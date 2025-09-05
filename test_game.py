from game import SimpleDrivingGame
import pygame

# Create the game
game = SimpleDrivingGame()

# Manual control loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Manual controls for testing
    keys = pygame.key.get_pressed()
    action = 0  # Default: do nothing
    
    if keys[pygame.K_LEFT]:
        action = 1
    elif keys[pygame.K_RIGHT]:
        action = 2
    elif keys[pygame.K_UP]:
        action = 3
    elif keys[pygame.K_DOWN]:
        action = 4
    
    # Step the game
    state, reward, done = game.step(action)
    
    if done:
        print(f"Game over! Distance traveled: {game.distance_traveled}")
        game.reset()
    
    # Render
    game.render()

pygame.quit()