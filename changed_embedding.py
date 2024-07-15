import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def load_scene(scene_id, dir):
    edges_file = os.path.join(dir, f"{scene_id}.edges")
    nodes_file = os.path.join(dir, f"{scene_id}.nodes")
    
    edges = pd.read_csv(edges_file, header=None, names=["target", "source"])
    edges = edges[(edges["target"] != -1) & (edges["source"] != -1)]
    nodes = pd.read_csv(nodes_file, header=None, na_values="_")
    valid_nodes = nodes.dropna(subset=[5, 6])
    return edges, valid_nodes

# Dataset
PATH = "dataset"
scene_ids = [fname.split('.')[0] for fname in os.listdir(PATH) if fname.endswith('.nodes')]

edges_list = []
nodes_list = []
for scene_id in scene_ids:
    edges, valid_nodes = load_scene(scene_id, PATH)
    edges_list.append(edges)
    nodes_list.append(valid_nodes)

all_edges = pd.concat(edges_list, ignore_index=True)
all_nodes = pd.concat(nodes_list, ignore_index=True)

# Encode node IDs
node_idx = {name: idx for idx, name in enumerate(sorted(all_nodes[0].unique()))}
all_nodes[0] = all_nodes[0].apply(lambda name: node_idx[name])
all_edges["source"] = all_edges["source"].apply(lambda name: node_idx.get(name, -1))
all_edges["target"] = all_edges["target"].apply(lambda name: node_idx.get(name, -1))
all_edges = all_edges[(all_edges["source"] != -1) & (all_edges["target"] != -1)]

node_features = all_nodes[[1, 2, 3, 4]].replace('_', np.nan).astype(float).to_numpy()
future_positions = all_nodes[[5, 6]].replace('_', np.nan).astype(float).to_numpy()
node_features = np.nan_to_num(node_features)
future_positions = np.nan_to_num(future_positions)

# Normalization
scaler_features = MinMaxScaler()
scaler_targets = MinMaxScaler()

node_features = scaler_features.fit_transform(node_features)
future_positions = scaler_targets.fit_transform(future_positions)

edges = tf.convert_to_tensor(all_edges[["target", "source"]], dtype=tf.int32)
node_states = tf.convert_to_tensor(node_features, dtype=tf.float32)

# train-val-test split
train_val_indices, test_indices, train_val_labels, test_labels = train_test_split(
    np.arange(len(node_features)), future_positions, test_size=0.2, random_state=SEED
)
train_indices, val_indices, train_labels, val_labels = train_test_split(
    train_val_indices, train_val_labels, test_size=0.125, random_state=SEED
)

class GraphAttention(layers.Layer):
    def __init__(self, units, kernel_initializer="glorot_uniform", kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel",
        )
        self.kernel_attention = self.add_weight(
            shape=(self.units * 2, 1),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_attention",
        )
        self.built = True

    def call(self, inputs):
        node_states, edges = inputs
        node_states_transformed = tf.matmul(node_states, self.kernel)
        node_states_expanded = tf.gather(node_states_transformed, edges)
        node_states_expanded = tf.reshape(node_states_expanded, (tf.shape(edges)[0], -1))
        attention_scores = tf.nn.leaky_relu(tf.matmul(node_states_expanded, self.kernel_attention))
        attention_scores = tf.squeeze(attention_scores, -1)
        attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))
        attention_scores_sum = tf.math.unsorted_segment_sum(
            data=attention_scores,
            segment_ids=edges[:, 0],
            num_segments=tf.reduce_max(edges[:, 0]) + 1,
        )
        attention_scores_sum = tf.repeat(
            attention_scores_sum, tf.math.bincount(tf.cast(edges[:, 0], "int32"))
        )
        attention_scores_norm = attention_scores / attention_scores_sum
        node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])
        out = tf.math.unsorted_segment_sum(
            data=node_states_neighbors * attention_scores_norm[:, tf.newaxis],
            segment_ids=edges[:, 0],
            num_segments=tf.shape(node_states)[0],
        )
        return out

class MultiHeadGraphAttention(layers.Layer):
    def __init__(self, units, num_heads=8, merge_type="concat", **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.merge_type = merge_type
        self.attention_layers = [GraphAttention(units) for _ in range(num_heads)]

    def call(self, inputs):
        atom_features, pair_indices = inputs
        outputs = [
            attention_layer([atom_features, pair_indices])
            for attention_layer in self.attention_layers
        ]
        if self.merge_type == "concat":
            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)
        return tf.nn.relu(outputs)

class GraphAttentionNetwork(keras.Model):
    def __init__(self, node_states, edges, hidden_units, num_heads, num_layers, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.node_states = node_states
        self.edges = edges
        self.preprocess = keras.Sequential([
            layers.Dense(hidden_units * num_heads, activation="relu"),
            layers.Dense(hidden_units * num_heads, activation="relu")
        ])
        self.attention_layers = [MultiHeadGraphAttention(hidden_units, num_heads) for _ in range(num_layers)]
        self.output_layer = layers.Dense(output_dim)

    def call(self, inputs):
        node_states, edges = inputs
        x = self.preprocess(node_states)
        for attention_layer in self.attention_layers:
            x = attention_layer([x, edges]) + x
        outputs = self.output_layer(x)
        return outputs

    def train_step(self, data):
        indices, labels = data
        with tf.GradientTape() as tape:
            outputs = self([self.node_states, self.edges])
            loss = self.compiled_loss(labels, tf.gather(outputs, indices))
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        indices = data
        outputs = self([self.node_states, self.edges])
        return tf.gather(outputs, indices)

    def test_step(self, data):
        indices, labels = data
        outputs = self([self.node_states, self.edges])
        loss = self.compiled_loss(labels, tf.gather(outputs, indices))
        self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))
        return {m.name: m.result() for m in self.metrics}

# Hyperparams
HIDDEN_UNITS = 100
NUM_LAYERS = 3
OUTPUT_DIM = 2
NUM_EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 1e-1
MOMENTUM = 0.9
# Grid search over the no. of attention heads
attention_heads_list = [4, 8, 16]

best_mse = float('inf')
best_heads = None

for num_heads in attention_heads_list:
    print(f"Training with {num_heads} attention heads...")
    
    # Recreate the optimizer to avoid the reuse error
    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.SGD(LEARNING_RATE, momentum=MOMENTUM, clipnorm=1.0)
    accuracy_fn = keras.metrics.MeanSquaredError(name="mse")
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_mse", min_delta=1e-5, patience=5, restore_best_weights=True
    )

    gat_model = GraphAttentionNetwork(
        node_states, edges, HIDDEN_UNITS, num_heads, NUM_LAYERS, OUTPUT_DIM
    )
    gat_model.compile(loss=loss_fn, optimizer=optimizer, metrics=[accuracy_fn])

    gat_model.fit(
        x=train_indices,
        y=train_labels,
        validation_data=(val_indices, val_labels),
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        callbacks=[early_stopping],
        verbose=2,
    )

    # Evaluate model
    _, val_mse = gat_model.evaluate(x=val_indices, y=val_labels, verbose=0)
    print(f"Validation Mean Squared Error with {num_heads} heads: {val_mse}")
    if val_mse < best_mse:
        best_mse = val_mse
        best_heads = num_heads

print("--" * 38 + f"\nBest number of attention heads: {best_heads} with MSE: {best_mse}")

# Evaluate on the test set
print(f"Evaluating best model with {best_heads} heads on the test set...")
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.SGD(LEARNING_RATE, momentum=MOMENTUM, clipnorm=1.0)
accuracy_fn = keras.metrics.MeanSquaredError(name="mse")

gat_model = GraphAttentionNetwork(
    node_states, edges, HIDDEN_UNITS, best_heads, NUM_LAYERS, OUTPUT_DIM
)
gat_model.compile(loss=loss_fn, optimizer=optimizer, metrics=[accuracy_fn])

gat_model.fit(
    x=np.concatenate([train_indices, val_indices]),
    y=np.concatenate([train_labels, val_labels]),
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    callbacks=[early_stopping],
    verbose=2,
)

_, test_mse = gat_model.evaluate(x=test_indices, y=test_labels, verbose=0)
print(f"Test Mean Squared Error: {test_mse}")

test_predictions = gat_model.predict(test_indices)
test_labels = scaler_targets.inverse_transform(test_labels)
test_predictions = scaler_targets.inverse_transform(test_predictions)

euclidean_distances = np.linalg.norm(test_labels - test_predictions, axis=1)
mean_euclidean_distance = np.mean(euclidean_distances)
print(f"Mean Euclidean Distance on the test set: {mean_euclidean_distance}")

'''
Best number of attention heads: 4 with MSE: 0.000137535942485556

4 attention head model on test set:
Test Mean Squared Error: 9.9326076451689e-05
Mean Euclidean Distance on the test set: 674.805150753234
'''