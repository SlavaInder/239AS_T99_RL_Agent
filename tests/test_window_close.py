import pygame
import time

pygame.init()
window = pygame.display.set_mode((200 , 200))
time.sleep(3)
print("closing down now")
pygame.display.quit()
pygame.quit() #Window doesn't close!
time.sleep(3) 