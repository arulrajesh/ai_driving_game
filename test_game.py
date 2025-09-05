from game import CheckpointGatesGame
import pygame

# Create the checkpoint gates game
game = CheckpointGatesGame()

# Manual control loop
running = True
print("Manual Controls:")
print("Arrow Keys: Left/Right to steer, Up to accelerate, Down to brake")
print("Goal: Drive through the numbered gates in sequence!")
print("Green gate = current target, White = future, Gray = completed")

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
    
    # Print feedback when gates are passed
    if reward > 50:  # Gate passed (gets 100 reward)
        print(f"Gate {game.gates_passed} passed! Reward: {reward:.1f}")
        print(f"Next target: Gate {game.current_gate + 1}")
    
    if done:
        if game.gates_passed >= game.total_gates:
            print(f"🎉 ALL GATES COMPLETED! Total gates: {game.gates_passed}")
        else:
            print(f"Time limit reached. Gates passed: {game.gates_passed}/{game.total_gates}")
        
        # Reset for another attempt
        print("Resetting for another attempt...")
        game.reset()
    
    # Render
    game.render()

pygame.quit()
print("Game ended!")