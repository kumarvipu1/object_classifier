# importing libraries
from tensorflow.keras import datasets, layers, models

# importing dataset
(training_img, training_labs), (test_img, test_labs) = datasets.cifar10.load_data()

# normalising dataset
training_img, test_img = training_img/255, test_img/255

# defining identification class
class_names = ['Plane', 'car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# visualizing the dataset (first 16 entries)
"""
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_img[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labs[i][0]]) 
"""

# reducing the size of dataset
training_img = training_img[:200000]
training_labs = training_labs[:200000]
test_img = test_img[:60000]
test_labs = test_labs[:60000]

# cnn model
model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_img, training_labs, epochs=10, validation_data=(test_img, test_labs))

loss, accuracy = model.evaluate(test_img, test_labs)
print(f"loss: {loss}")
print(f"accuracy: {accuracy}")

model.save("image_classifier.model")

