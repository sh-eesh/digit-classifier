
import pygame
import sys
import torch
from PIL import Image
from torchvision import transforms

from models import DigitClassifier

# Initialize Pygame
pygame.init()

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load trained model
model = DigitClassifier().to(device)

model.load_state_dict(
    torch.load(
        "checkpoints/model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

# Window settings
WIDTH = 500
HEIGHT = 580

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Digit Classifier")

# Colors
BACKGROUND = (20, 22, 28)
CANVAS_COLOR = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (145, 150, 160)
BUTTON_COLOR = (45, 49, 60)
BUTTON_HOVER = (65, 70, 85)
BLUE = (55, 110, 220)

# Fonts
title_font = pygame.font.SysFont("segoeui", 30, bold=True)
normal_font = pygame.font.SysFont("segoeui", 16)
small_font = pygame.font.SysFont("segoeui", 13)

# Drawing canvas
CANVAS_SIZE = 280
CANVAS_X = (WIDTH - CANVAS_SIZE) // 2
CANVAS_Y = 130

canvas = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
canvas.fill(CANVAS_COLOR)

# Drawing settings
brush_size = 15
drawing = False
last_pos = None

# Buttons
clear_button = pygame.Rect(110, 455, 130, 45)
predict_button = pygame.Rect(260, 455, 130, 45)

# Prediction result
prediction = None
confidence = None

# Image transformation
transform = transforms.ToTensor()

# Main clock
clock = pygame.time.Clock()


# Draw centered text
def draw_text(text, font, color, y):
    surface = font.render(text, True, color)
    rect = surface.get_rect(center=(WIDTH // 2, y))
    screen.blit(surface, rect)


# Clear the drawing canvas
def clear_canvas():
    global prediction, confidence

    canvas.fill(CANVAS_COLOR)

    prediction = None
    confidence = None


# Preprocess image to match MNIST format
def preprocess_image(image):

    # Convert to grayscale
    image = image.convert("L")

    # Find digit boundaries
    bbox = image.getbbox()

    if bbox is None:
        return None

    # Crop around digit
    image = image.crop(bbox)

    # Preserve aspect ratio
    width, height = image.size
    size = max(width, height)

    square = Image.new("L", (size, size), 0)

    x = (size - width) // 2
    y = (size - height) // 2

    square.paste(image, (x, y))

    # Resize digit to 20x20
    image = square.resize(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    # Center inside 28x28 image
    final_image = Image.new("L", (28, 28), 0)
    final_image.paste(image, (4, 4))

    # Convert to tensor
    return transform(final_image)


# Predict the digit
def predict_digit():
    global prediction, confidence

    # Convert Pygame canvas into a PIL image
    image_string = pygame.image.tobytes(canvas, "RGB")

    image = Image.frombytes(
        "RGB",
        canvas.get_size(),
        image_string
    )

    # Preprocess image
    image = preprocess_image(image)

    if image is None:
        prediction = None
        confidence = None
        return

    # Add batch dimension and move to device
    image = image.unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence_tensor, prediction_tensor = probabilities.max(dim=1)

    # Extract Python values
    prediction = prediction_tensor.item()
    confidence = confidence_tensor.item() * 100


# Main application loop
running = True

while running:
    mouse_pos = pygame.mouse.get_pos()

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Start drawing
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:

                if clear_button.collidepoint(event.pos):
                    clear_canvas()

                elif predict_button.collidepoint(event.pos):
                    predict_digit()

                elif pygame.Rect(
                    CANVAS_X,
                    CANVAS_Y,
                    CANVAS_SIZE,
                    CANVAS_SIZE
                ).collidepoint(event.pos):

                    drawing = True

                    x = event.pos[0] - CANVAS_X
                    y = event.pos[1] - CANVAS_Y

                    last_pos = (x, y)

                    pygame.draw.circle(
                        canvas,
                        WHITE,
                        last_pos,
                        brush_size // 2
                    )

        # Stop drawing
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False
                last_pos = None

        # Draw continuous lines
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                x = event.pos[0] - CANVAS_X
                y = event.pos[1] - CANVAS_Y

                current_pos = (x, y)

                if 0 <= x < CANVAS_SIZE and 0 <= y < CANVAS_SIZE:

                    pygame.draw.line(
                        canvas,
                        WHITE,
                        last_pos,
                        current_pos,
                        brush_size
                    )

                    pygame.draw.circle(
                        canvas,
                        WHITE,
                        current_pos,
                        brush_size // 2
                    )

                    last_pos = current_pos

    # Draw background
    screen.fill(BACKGROUND)

    # Application heading
    draw_text("Digit Classifier", title_font, WHITE, 45)

    draw_text(
        "Draw a handwritten digit from 0 to 9",
        normal_font,
        GRAY,
        85
    )

    # Draw canvas
    screen.blit(canvas, (CANVAS_X, CANVAS_Y))

    pygame.draw.rect(
        screen,
        (65, 68, 78),
        (CANVAS_X, CANVAS_Y, CANVAS_SIZE, CANVAS_SIZE),
        2
    )

    # Draw Clear button
    clear_color = (
        BUTTON_HOVER
        if clear_button.collidepoint(mouse_pos)
        else BUTTON_COLOR
    )

    pygame.draw.rect(
        screen,
        clear_color,
        clear_button,
        border_radius=8
    )

    clear_text = normal_font.render("Clear", True, WHITE)
    screen.blit(clear_text, clear_text.get_rect(center=clear_button.center))

    # Draw Predict button
    predict_color = (
        (75, 135, 245)
        if predict_button.collidepoint(mouse_pos)
        else BLUE
    )

    pygame.draw.rect(
        screen,
        predict_color,
        predict_button,
        border_radius=8
    )

    predict_text = normal_font.render("Predict", True, WHITE)
    screen.blit(
        predict_text,
        predict_text.get_rect(center=predict_button.center)
    )

    # Display prediction
    if prediction is not None:
        draw_text(
            f"Predicted digit: {prediction}",
            normal_font,
            WHITE,
            525
        )

        draw_text(
            f"Confidence: {confidence:.2f}%",
            small_font,
            GRAY,
            550
        )

    else:
        draw_text(
            "Draw a digit and press Predict",
            small_font,
            GRAY,
            535
        )

    # Update display
    pygame.display.flip()

    clock.tick(60)


# Close application
pygame.quit()
sys.exit()
