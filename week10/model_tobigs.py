import os
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        self.dense1 = layers.Dense(7*7*256, use_bias= False, input_shape=(100,))
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.LeakyReLU(alpha = 0.2) #논문에 나온대로 기울기(0.2) 설정

        self.reshape = layers.Reshape((7,7,256))

        self.convT1 = layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding = 'same', use_bias = False)
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.LeakyReLU(alpha = 0.2) #논문에 나온대로 기울기(0.2) 설정

        self.convT2 = layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding = 'same', use_bias = False)
        self.bn3 = layers.BatchNormalization()
        self.relu3 = layers.LeakyReLU(alpha = 0.2) #논문에 나온대로 기울기(0.2) 설정

        self.convT3 = layers.Conv2DTranspose(1, (5,5), strides = (2,2), padding = 'same', use_bias = False, activation = 'tanh')
        #논문에 나온대로 생성자의 마지막의 활성화 함수 tanh 사용

    def call(self, input): #텐서플로우가 미숙해서 Sequential이용을 못 했습니다
        x = self.dense1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.reshape(x)
        x = self.convT1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.convT2(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.convT3(x)

        return x



class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = layers.Conv2D(64, (5,5), strides = (2,2), padding = 'same', input_shape = [28,28,1])
        self.relu1 = layers.LeakyReLU(alpha = 0.2) #논문에 나온대로 기울기(0.2) 설정
        self.drop1 = layers.Dropout(0.3)

        self.conv2 = layers.Conv2D(128, (5,5), strides = (2,2), padding = 'same')
        self.relu2 = layers.LeakyReLU(alpha = 0.2) #논문에 나온대로 기울기(0.2) 설정
        self.drop2 = layers.Dropout(0.3)

        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1) #binarycrossentropy 이므로 출력 1

    def call(self, input):
        x = self.conv1(input)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.flatten(x)
        x = self.dense(x)

        return x

def discriminator_loss(loss_object, real_output, fake_output):
    #here = tf.ones_like(????) or tf.zeros_like(????)  -> tf.zeros_like와 tf.ones_like에서 선택하고 (???)채워주세요
    #진짜일 경우 1이므로 ones_like사용해서 loss 계산
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    #가짜일 경우 0이므로 zeoos_like사용해서 loss 계산
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss #판별자의 경우 두 loss를 더한다
    return total_loss


def generator_loss(loss_object, fake_output):
    #생성자의 경우 판별자가 1로 판별하는 fake_ouput을 만들어야 하므로 ones_like를 이용해서 loss를 줄여간다
    return loss_object(tf.ones_like(fake_output), fake_output)

def normalize(x):
    image = tf.cast(x['image'], tf.float32)
    image = (image / 127.5) - 1
    return image


def save_imgs(epoch, generator, noise):
    gen_imgs = generator(noise, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(gen_imgs.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(gen_imgs[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    fig.savefig("images/mnist_%d.png" % epoch)

def train():
    data, info = tfds.load("mnist", with_info=True, data_dir='data/tensorflow_datasets')
    train_data = data['train']

    if not os.path.exists('./images'):
        os.makedirs('./images')

    # settting hyperparameter
    latent_dim = 100 #논문에서 dim(z) = (100,)
    epochs = 2
    batch_size = 10000
    buffer_size = 6000
    save_interval = 1

    generator = Generator()
    discriminator = Discriminator()

    #따로 훈련되므로 각각의 optimizer 필요
    #논문에 나온대로 learning_rate를 0.001->0.0002, beta_1을 0.9 -> 0.5
    gen_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1 = 0.5, beta_2 = 0.999)
    disc_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1 = 0.5, beta_2 = 0.999)

    train_dataset = train_data.map(normalize).shuffle(buffer_size).batch(batch_size)

    #손실함수 = binarycrossentropy
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)

    @tf.function
    def train_step(images):
        noise = tf.random.normal([batch_size, latent_dim])

        with tf.GradientTape(persistent=True) as tape:
            generated_images = generator(noise, training = True) #이미지 생성

            real_output = discriminator(images, training = True) #판별자가 진짜 이미지 판별
            generated_output = discriminator(generated_images, training = True) #판별자가 가짜 이미지 판별

            gen_loss = generator_loss(cross_entropy, generated_output) #생성자 loss 계산
            disc_loss = discriminator_loss(cross_entropy, real_output, generated_output) #판별자 loss 계산

        #이후 역전파 과정 수행
        grad_gen = tape.gradient(gen_loss, generator.trainable_variables)
        grad_disc = tape.gradient(disc_loss, discriminator.trainable_variables)

        gen_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))

        return gen_loss, disc_loss

    seed = tf.random.normal([16, latent_dim])

    for epoch in range(epochs):
        start = time.time()
        total_gen_loss = 0
        total_disc_loss = 0

        for images in train_dataset:
            gen_loss, disc_loss = train_step(images)

            total_gen_loss += gen_loss
            total_disc_loss += disc_loss

        print('Time for epoch {} is {} sec - gen_loss = {}, disc_loss = {}'.format(epoch + 1, time.time() - start, total_gen_loss / batch_size, total_disc_loss / batch_size))
        if epoch % save_interval == 0:
            save_imgs(epoch, generator, seed)


if __name__ == "__main__":
    train()
