from game_displayer import basicdisplayer
def main():
    red_player_count = 1
    blue_player_count = 1
    player_count = red_player_count + blue_player_count
    ball_count = 1
    disp = basicdisplayer.GameWindow(1096, 400)
    running = True
    while(running):
        disp.drawThings(game.getState("full info"))
        disp.clock.tick(disp.fps)
        pygame.display.update()














if __name__ == "__main__":
    main()
