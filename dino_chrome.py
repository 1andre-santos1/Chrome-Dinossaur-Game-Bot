import cv2
from PIL import ImageGrab
import numpy as np
import pyautogui
import time

# does the console interaction with the user
def InteractWithUser():
    windowedInput = str(raw_input("Do you want to visualize the objects detection? (y/n)\n"))
    windowed = True
    if windowedInput == "n":
        windowed = False

    print("The app has started, load any of the versions of the chrome dinossaur game.")
    print("\nNight Mode is supported!")
    print("\nPress the 'e' key on the window for object detection visualization to quit the app or close this console.")
    return windowed

# starts the bot
def StartBot(windowed):
    
    # dimensions and top left position of the chrome dinossaur game
    # (to limit the pixels for the object detection process)
    game_width = 600
    game_height = 200
    game_leftX = 0
    game_topY = 0
    
    # maximum distance that a detected object could be to trigger a jump
    max_distance = 150
    # value of the max distance increment between frames
    # (as the velocity of the world increases, a greater 'max_distance' is needed)
    inc_max_distance = 0.03

    # boolean that is True when night, and vice versa
    night = False

    # images of the objects needed for the app
    # (dinossaur,the two types of cactus,bird and the game over button)
    dino = cv2.imread('dino.png',0)
    cactus = cv2.imread('cactus.png',0)  
    cactus2 = cv2.imread('cactus02.png',0)
    bird = cv2.imread('bird.png',0)
    btn_gameOver = cv2.imread('btn_gameOver.png',0)

    # app loop
    while True:
        # if the top left position of the screen is not assigned
        if game_leftX == 0:
            # the color image to analyze will be the entire screen
            screen_img_rgb = ImageGrab.grab()
        else:
            # the color image to analyze will be, only, the box where the dinossaur game is played
            screen_img_rgb = ImageGrab.grab(bbox = (game_leftX,game_topY,game_leftX + game_width, game_topY + game_height))
            
        # convert color image into gray and store it in a variable
        screen_img_gray = cv2.cvtColor(np.array(screen_img_rgb), cv2.COLOR_BGR2GRAY)

        # get the average color of the pixels in the gray image
        # (close to 0(black) -> night || close to 255(white) -> day)
        avg_color_per_row = np.average(screen_img_gray, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)

        # if the average color is below or equal to 50 and wasn't night
        if avg_color <= 50 and night is False:
            # invert the color of all the images needed for detection
            # (the original black images will be inverted to white)
            dino = cv2.bitwise_not(dino)
            cactus = cv2.bitwise_not(cactus)  
            cactus2 = cv2.bitwise_not(cactus2)
            bird = cv2.bitwise_not(bird)
            btn_gameOver = cv2.bitwise_not(btn_gameOver)
            # set night to True
            night = True
        # if the average color is greater than 50 and was night
        elif avg_color > 50 and night is True:
            # read the original black images
            dino = cv2.imread('dino.png',0)
            cactus = cv2.imread('cactus.png',0)  
            cactus2 = cv2.imread('cactus02.png',0)
            bird = cv2.imread('bird.png',0)
            btn_gameOver = cv2.imread('btn_gameOver.png',0)
            # set night to False
            night = False

        # get the dimensions of all images needed for detection
        dino_width,dino_height = dino.shape[::-1]
        cactus_width, cactus_height = cactus.shape[::-1]
        cactus2_width, cactus2_height = cactus2.shape[::-1]
        bird_width,bird_height = bird.shape[::-1]
        btn_gameOver_width,btn_gameOver_height = bird.shape[::-1]

        # detect the images on the gray image
        res_dino = cv2.matchTemplate(screen_img_gray,dino,cv2.TM_CCOEFF_NORMED)
        rescactus = cv2.matchTemplate(screen_img_gray,cactus,cv2.TM_CCOEFF_NORMED)
        rescactus2 = cv2.matchTemplate(screen_img_gray,cactus2,cv2.TM_CCOEFF_NORMED)
        resbird = cv2.matchTemplate(screen_img_gray,bird,cv2.TM_CCOEFF_NORMED)
        resbtn_gameOver = cv2.matchTemplate(screen_img_gray,btn_gameOver,cv2.TM_CCOEFF_NORMED)

        # threshold of all the images to detect
        # (certainty percentage of the matching result)
        threshold_dino = 0.75
        threshold_cactus = 0.6                
        threshold_cactus2 = 0.7
        threshold_bird = 0.6
        threshold_btn_gameOver = 0.8

        # get an array with the locations of the detections for every image
        # (where the matching result has a certainty greater or equal to the threshold)
        location_dino = np.where(res_dino >= threshold_dino)
        location_cactus = np.where(rescactus >= threshold_cactus)
        location_cactus2 = np.where(rescactus2 >= threshold_cactus2)
        location_bird = np.where(resbird >= threshold_bird)
        location_btn_gameOver = np.where(resbtn_gameOver >= threshold_btn_gameOver)

        # the dinossaur position is set to 0
        dino_x = 0
        dino_y = 0
        # for every dinossaur detected
        for pt in zip(*location_dino[::-1]):
            # set the dinossaur position to where it was detected
            dino_x = pt[0]
            dino_y = pt[1]
            # exit this loop
            # (there's only one dinossaur)
            break

        # if the left position of the screen is not assigned
        if (game_leftX == 0):
            # set the position of the screen to the game bounds
            game_leftX = dino_x
            game_topY = dino_y - 100

        # the dinossaur didn't jump
        hasJumped = False
        # for every cactus, type 1, detection
        for pt in zip(*location_cactus[::-1]):
            # if the dinossaur has jumped
            if(hasJumped is True):
                # exit this loop
                break
            # get this cactus position
            cactus_x = pt[0]
            cactus_y = pt[1]
            # if the window for the detection visualization is active
            if windowed:
                # if it is night
                if night:
                    # draw a white box arround this object
                    cv2.rectangle(screen_img_gray,(pt[0],pt[1]),(pt[0]+cactus_width,pt[1]+cactus_height),(255,255,255),1)
                else:
                    # draw a black box arround this object
                    cv2.rectangle(screen_img_gray,(pt[0],pt[1]),(pt[0]+cactus_width,pt[1]+cactus_height),(0,0,0),1)
            # if the distance from this cactus to the dinossaur is below the max distance and the cactus is on the right of the dinossaur
            if (cactus_x > dino_x) and (cactus_x - dino_x <= max_distance):
                # press the space key (jump)
                pyautogui.press('space')
                # the dinossaur has jumped
                hasJumped = True
                # exit this loop
                break
            
        # if the dinossaur didn't jump       
        if hasJumped is not True:
            # for every cactus, type 2, detection
            for pt in zip(*location_cactus2[::-1]):
                # if the dinossaur has jumped
                if(hasJumped is True):
                    # exit this loop
                    break
                # get this cactus position
                cactus2_x = pt[0]
                cactus2_y = pt[1]
                # if the window for the detection visualization is active
                if windowed:
                    # if it is night
                    if night:
                        # draw a white box arround this object
                        cv2.rectangle(screen_img_gray,(pt[0],pt[1]),(pt[0]+cactus2_width,pt[1]+cactus2_height),(255,255,255),1)
                    else:
                        # draw a black box arround this object
                        cv2.rectangle(screen_img_gray,(pt[0],pt[1]),(pt[0]+cactus2_width,pt[1]+cactus2_height),(0,0,0),1)
                # if the distance from this cactus to the dinossaur is below the max distance and the cactus is on the right of the dinossaur
                if (cactus2_x > dino_x) and (cactus2_x - dino_x <= max_distance):
                    # press the space key (jump)
                    pyautogui.press('space')
                    # the dinossaur has jumped
                    hasJumped = True
                    # exit this loop
                    break
                
        # if the dinossaur didn't jump 
        if hasJumped is not True:
            # for every bird detection
            for pt in zip( *location_bird[::-1]):
                # if the dinossaur has jumped
                if(hasJumped is True):
                    # exit this loop
                    break
                # get this bird position
                bird_x = pt[0]
                bird_y = pt[1]
                # if the window for the detection visualization is active
                if windowed:
                    # if it is night
                    if night:
                        # draw a white box arround this object
                        cv2.rectangle(screen_img_gray,(pt[0],pt[1]),(pt[0]+bird_width,pt[1]+bird_height),(255,255,255),1)
                    else:
                        # draw a black box arround this object
                        cv2.rectangle(screen_img_gray,(pt[0],pt[1]),(pt[0]+bird_width,pt[1]+bird_height),(0,0,0),1)
                # if the distance from this cactus to the dinossaur is below the max distance and the cactus is on the right of the dinossaur
                if (bird_x > dino_x) and (bird_x - dino_x <= max_distance):
                    # if the bird is above the dinossaur
                    if ((bird_y + bird_height) <= (dino_y)):
                        # exit this loop
                        break
                    # if the bird is on below the dinossaur head but above the middle of the dinossaur
                    elif ((bird_y + bird_height) <= (dino_y + (dino_height/2))):
                        # press down the down arrow key (crouch)
                        pyautogui.keyDown('down')
                        # wait 250ms 
                        time.sleep(0.25)
                        # release the down arrow key
                        pyautogui.keyUp('down')
                    # if the bird is below the middle of the dinossaur
                    else:
                        # press the space key (jump)
                        pyautogui.press('space')
                    # the dinossaur has jumped
                    hasJumped = True
                    # exit the loop
                    break

        # increment the max distance
        max_distance += inc_max_distance
        # fix the max distance (if greater than) to the game width/3
        if max_distance >= game_width / 3:
            max_distance = game_width/3

        # if the dinossaur didn't jump
        if hasJumped is not True:
            # for every game over button detection
            for pt in zip(*location_btn_gameOver[::-1]):
                # if the dinossaur has jumped 
                if(hasJumped is True):
                    # exit loop
                    break
                # get the button position
                btn_gameOver_x = pt[0]
                btn_gameOver_y = pt[1]
                # restart the max distance
                max_distance = 150
                # press the space button
                # to restart the game
                pyautogui.press('space')
                # set has jumped to true
                hasJumped = True
                # exit this loop
                break  
        # if the window for the detection visualization is active 
        if windowed:
            # show the resulting gray image
            cv2.imshow('Chrome Dinossaur - Bot',screen_img_gray)
            # resize the window to the game bounds
            cv2.resizeWindow('Chrome Dinossaur - Bot',game_width,game_height)
            # if the user press the 'e' key
            if cv2.waitKey(1) & 0xFF == ord('e'):
                # destroy all windows
                cv2.destroyAllWindows()
                # exit the main loop
                break

windowed = InteractWithUser()
StartBot(windowed)

