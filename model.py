import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt  # Needed for saving images
import numpy as np

# Define paths for saving models and images
checkpoint_dir = './training_checkpoints'
image_dir = './generated_images'
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

# Data preprocessing
data_gen = ImageDataGenerator(rescale=1./255)
train_data = data_gen.flow_from_directory(
    'dataset\dataset',  # This is the updated path
    target_size=(32, 32),
    batch_size=64,
    class_mode='categorical',
    color_mode='grayscale'
)


# Generator Model
from tensorflow.keras import layers

def make_dcgan_generator(z_dim):
    model = tf.keras.Sequential([
       
        layers.Dense(4*4*256, use_bias=False, input_shape=(z_dim,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((4, 4, 256)),
        
        # Upsample to 8x8
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        # Upsample to 16x16
        layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh'),
        layers.Reshape((32, 32, 1)), # Add a reshape layer to add the channel dimension
 
    ])
    return model


def make_dcgan_discriminator(num_emotions=5):
    # Input layer
    inp = layers.Input(shape=(32, 32, 1))

    x = layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')(inp)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)

    # Real/Fake output
    out1 = layers.Dense(1)(x)

    # Emotion classification output
    out2 = layers.Dense(num_emotions, activation='softmax')(x)

    # Updated model with two outputs
    model = tf.keras.Model(inp, [out1, out2])

    return model




# Loss Functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output, real_emotions, predicted_emotions):
    # Binary cross-entropy for real/fake classification
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    rf_loss = real_loss + fake_loss

    # Categorical cross-entropy for emotion classification
    emotion_loss = tf.keras.losses.categorical_crossentropy(real_emotions, predicted_emotions)

    total_loss = rf_loss + emotion_loss  # Combine losses
    return total_loss




# Generator and discriminator models
generator = make_dcgan_generator(z_dim=100)
discriminator = make_dcgan_discriminator()

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


@tf.function
def train_step(images, labels):
    noise_dim = 100
    batch_size = tf.shape(images)[0]
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output, real_emotions = discriminator(images, training=True)
        generated_output, predicted_emotions = discriminator(generated_images, training=True)
        gen_loss = generator_loss(generated_output)
        disc_loss = discriminator_loss(real_output, generated_output, real_emotions, predicted_emotions)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

# Training loop with progress updates, checkpointing, and image saving
def train(dataset, epochs, num_examples_to_generate=5, max_batches=100):
    noise_dim = 100  # Define noise_dim here

    for epoch in range(epochs):
        gen_loss_list = []
        disc_loss_list = []

        for batch_number, (image_batch, labels_batch) in enumerate(dataset):
            if batch_number >= max_batches:
                break

            gen_loss, disc_loss = train_step(image_batch, labels_batch)
            gen_loss_list.append(gen_loss)
            disc_loss_list.append(disc_loss)

            if batch_number % 100 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_number}, Gen Loss: {gen_loss}, Disc Loss: {disc_loss}')

        # Save the model every epoch
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        generator.save_weights(checkpoint_prefix + '_generator.weights.h5')
        discriminator.save_weights(checkpoint_prefix + '_discriminator.weights.h5')

        # Generate and save images after each epoch
        seed = tf.random.normal([num_examples_to_generate, noise_dim])
        predictions = generator(seed, training=False)
        save_images(predictions, epoch + 1, image_dir)

        print(f'Epoch {epoch + 1}/{epochs} completed')



# Function to save generated images
def save_images(predictions, epoch, image_dir):
    # Determine the square size of the subplot grid based on the number of images
    subplot_square = int(np.ceil(np.sqrt(predictions.shape[0])))

    fig = plt.figure(figsize=(subplot_square, subplot_square))  # Adjust figsize if needed

    for i in range(predictions.shape[0]):
        plt.subplot(subplot_square, subplot_square, i + 1)
        # Reshape the image to remove the last dimension and scale pixel values back to [0,1]
        image = np.reshape(predictions[i], (32, 32)) * 0.5 + 0.5
        plt.imshow(image, cmap='gray')  # Show the image in grayscale
        plt.axis('off')

    # Ensure the image directory exists
    os.makedirs(image_dir, exist_ok=True)
    # Save the figure
    plt.savefig(os.path.join(image_dir, f'image_at_epoch_{epoch:04d}.png'))
    plt.close()


# Start Training
train(train_data, epochs=15)

discriminator.save('discriminator_model.h5')
