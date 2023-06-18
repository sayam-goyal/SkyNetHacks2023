import cv2 as cv
import numpy as np
import os
import face_recognition as fr

def getFaces(img):
    faces = [np.load("People/"+dir) for dir in os.listdir("People")]
    names = [os.path.splitext(dir)[0] for dir in os.listdir("People")]
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    locs = fr.face_locations(img)
    encode = fr.face_encodings(img)
    people = []
    unkwown = []
    for enc, loc in zip(encode, locs):
        y1, x2, y2, x1 = loc
        dis = fr.face_distance(faces, enc)
        min = 0
        for i in range(len(dis)):
            if dis[i] < dis[min]:
                min = i
        if len(dis) != 0 and dis[min] < 0.5:
            people.append(names[min])
            del faces[min]
            del names[min]
        else:
            unkwown.append([(x1, y1, x2, y2), enc])
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    return people, unkwown, names

cam = cv.VideoCapture(2)
cv.namedWindow("Camera")
days = 0
while True:
    ret, img = cam.read()
    if not ret:
        break
    k = cv.waitKey(1)%256
    cv.imshow("Camera", img)
    if k == 27:
        break
    elif k == 32:
        people, unkwown, absent = getFaces(img)
        for i in range(len(people)):
            people[i]+=" is here"
        for person in absent:
            people.append(person+" is not here")
        if len(unkwown) > 0:
            cv.namedWindow("Unkwown")
            for person in unkwown:
                x1, y1, x2, y2 = person[0]
                cv.imshow("Unkwown", img[y1:y2, x1:x2])
                name = ""
                print("Who is this")
                while True:
                    k = cv.waitKey(0)%256
                    if k == 225:
                        continue
                    if k == 13:
                        break
                    if k == 8:
                        name = name[:-1]
                        continue
                    k = chr(k)
                    if k.islower() or k.isupper():
                        name+=k
                name = name.title()
                people.append(name+" is here")
                np.save(f"People/{name}.npy", person[1])
            cv.destroyWindow("Unkwown")
        sorted(people)
        days+=1
        print("Day "+str(days)+": ")
        for inx, person in enumerate(people):
            print("\t"+str(inx+1)+". "+person)
        print()
cv.destroyWindow("Camera")