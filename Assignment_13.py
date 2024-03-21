import numpy as np

def generate_images_and_filters(num_images, num_filters):
    images = []
    filters = []

    # Generate random images
    for _ in range(num_images):
        image = np.random.randint(0, 256, size=(6, 6))  # Random integers between 0 and 255
        images.append(image)

    # Generate random filters
    for _ in range(num_filters):
        filter = np.random.randint(-1, 2, size=(3, 3))  # Random integers between -1 and 1
        filters.append(filter)

    return images, filters

def convolve_images(images, filters):
    output_images = []

    for image in images:
        for filter in filters:
            output_dim = image.shape[0] - filter.shape[0] + 1
            output = np.zeros((output_dim, output_dim))

            for i in range(output_dim):
                for j in range(output_dim):
                    output[i, j] = np.sum(image[i:i+3, j:j+3] * filter)

            output_images.append(output)

    return output_images

# Example usage
num_images = 5
num_filters = 3
images, filters = generate_images_and_filters(num_images, num_filters)

# Perform convolution
convolved_images = convolve_images(images, filters)

# Print the generated images, filters, and convolved images
print("Generated Images:")
for i, image in enumerate(images):
    print(f"Image {i+1}:\n{image}\n")

print("Generated Filters:")
for i, filter in enumerate(filters):
    print(f"Filter {i+1}:\n{filter}\n")

print("Convolved Images:")
for i, convolved_image in enumerate(convolved_images):
    print(f"Convolved Image {i+1}:\n{convolved_image}\n")
