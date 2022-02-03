'''This is a helper script to facilitate convenient data collection for building KG CV capability.

This script should be used to capture every move of a chess game (from a single fixed camera angle)
as specified by a PGN that gives the sequence of moves played. The output of this script is a
named folder (name provided by user) of images, one per move of the game. A corresponding PGN with
should be placed in this folder after collecting the images. The images and PGN can then be used
with the `extract_labels_from_game.py` script, also in this directory.

Note: Images should be captured s.t. `a1` corresponds to bottom left corner of board in image.
'''

import os

import cv2

def main():
    '''Main loop to capture images from user keyboard input.
    '''
    game_name = None
    # Prompt user for game name until they provide a dir name that does not already exist.
    while not game_name:
        game_name = input("Please name this game: ")
        try:
            os.mkdir(game_name)
        except FileExistsError as _:
            print(f"The directory {game_name} already exists, please try again.")
            game_name = None

    # TODO: update this to use correct camera index on raspi
    camera_idx = 0

    # Set up camera
    cam = cv2.VideoCapture(camera_idx)
    window_name = "KG Data Collection"
    cv2.namedWindow(window_name)
    move_counter = 0

    # Main loop, captures images when SPACE pressed
    while True:
        ret, frame = cam.read()

        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1)

        if key % 256 == 27:
            # ESC pressed
            print("Escape hit, ending data collection...")
            break
        if key % 256 == 32:
            # SPACE pressed
            img_name = f"{game_name}/move_{move_counter}.png"
            cv2.imwrite(img_name, frame)
            print(f"Wrote move {move_counter}")
            move_counter += 1

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
