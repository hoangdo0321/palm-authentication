from extract import *

if __name__ == '__main__':
    #####
    path_in_img = "E:/Palm-Authentication/train_images/Hoang/raw/Hoang_13.jpg"

    extract =  Extract()

    extract.extract_roi(path_in_img, rotate=True)
    extract.show_result()
    extract.save("resources/ex10.jpg")