import cv2
import numpy as np


cap = cv2.VideoCapture(cv2.CAP_AVFOUNDATION)
# 0 means using the 0th camera. Put PATH here if loading in video file
cap.set(3, 720)
cap.set(4, 480)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fourcc is probably a code indicating which video compressor to use. Note that here *'mp4v' == 'm','p','4','v'
out = cv2.VideoWriter('output.mov', fourcc, 20.0,
                      (int(cap.get(3)), int(cap.get(4))))
# 20.0 indicates the fps which pretty much dictates the playback speed
# cap.get(3) provides the width of the input video and cap.get(4) provides the height. For Mac specifically, the dimensions specified in the writer has to match exactly the dimension you actually put in (makes sense)
ret, frame = cap.read()
cv2.startWindowThread()
while ret:
    ret, frame = cap.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # notice that what is written in is colored frames (alpha channel removed (camera doesn't necessarily support it?))
    cv2.line(frame, (0, 0), (150, 150), (230, 40, 100), thickness=5)
    # cv2.line is a function which modifies the numpy frame; first argument is the start, second the end, third color, fourth linewidth
    cv2.rectangle(grayFrame, (23, 340), (234, 445),
                  color=(190, 255, 255), thickness=5)
    # cv2.rectangle is similar. First is upper left, second is lower right. Of course only the first color channel matters and you only need to put one number (controlling grayness)
    cv2.circle(frame, (100, 300), 50, color=(32, 43, 123), thickness=-1)
    # -1 thickness means fill, others are intuitive
    points = np.linspace(10, 120, 12, dtype=np.int32)
    # we have to use integer because you can't plot a point on a float-valued pixel position
    np.random.shuffle(points)
    points = points.reshape((6, 2))
    cv2.polylines(grayFrame, [points, points+200],
                  True, color=123, thickness=5)
    # it takes in a list of nparrays of points. For each element in the list, it plots a polygon for it. Second argument controls whether the polygon closes or not.

    cv2.putText(frame, "hello idiot", (34, 123), fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=5, color=(123, 12, 43), thickness=5)

    # remember all these above operations are operated on np arrays and not yet rendered onto the screen (much like in matplotlib where we do the work in the background before rendering it up). The rendering is done by imshow

    # Common np operations are of course still applicable
    frame[230:330, 230:330, 0] = grayFrame[230:330, 230:330]
    frame[230:330, 230:330, 1] = grayFrame[230:330, 230:330]
    frame[230:330, 230:330, 2] = grayFrame[230:330, 230:330]

    out.write(frame)
    cv2.imshow('frame', frame)
    cv2.imshow('gray', grayFrame)
    # imshow must be used in coalition with cv2.waitKey or a similar command because imshow only shows image for a split second and will not wait if there's no waitKey command. WaitKey command waits for the specified number of miliseconds while reading in the key (this is the time during which that image will be showed). Returns -1 if no key is read. If the input is 0, waitKey waits indefinitely.

    # color filtering (basically to mask out all colors except a specific range of colors)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # hsv encoding means hue, saturation, value. Basically instead of composing color from rgb, hue gives a spectrum of color similar to light. Saturation determines the intensity of the color. Value determines brightness.
    lowerGreen = np.array([30, 20, 0])
    upperGreen = np.array([80, 255, 180])
    # this of course returns 255 or 0. Note that 255 in bitwise is basically 1 since all bits are 1
    greenMask = cv2.inRange(hsv, lowerGreen, upperGreen)
    cv2.imshow('green mask', greenMask)

    # noise removal by Morphological Transformations
    # erosion
    kernel2 = np.ones((5, 5), dtype=np.uint8)/25
    eroded = cv2.erode(greenMask, kernel2, iterations=1)
    # erosion is often used for grayscale. In colored image, each channel is eroded independently. Iteration means how many times erosion is applied. Erosion basically makes a pixel 0 if there exist a pixel in its kernel that isn't 1. So expectedly, it will clear white noise.

    # dilation
    dilated = cv2.dilate(greenMask, kernel2, iterations=1)
    # dilation is opposite. It clears black noise by making a pixel 1 if there exist a pixel in its kernel that isn't 0

    # opening = erosion followed by dilation
    opened = cv2.morphologyEx(greenMask, cv2.MORPH_OPEN, kernel2, iterations=1)
    # obviously opening magnifies black since erosion is placed first. The second dilation cannot compensate for the small white pixels eroded
    # closing = dilation followed by erosion
    closed = cv2.morphologyEx(
        greenMask, cv2.MORPH_CLOSE, kernel2, iterations=1)
    # in contrast closing magnifies white

    cv2.imshow('eroded green mask', eroded)
    cv2.imshow('dilated green mask', dilated)
    cv2.imshow('opened green mask', opened)
    cv2.imshow('closed green mask', closed)
    greenPixels = cv2.bitwise_and(frame, frame, mask=greenMask)
    cv2.imshow('green parts', greenPixels)

    # blurring and smoothing
    # normal filter
    kernel = np.ones((15, 15), dtype=np.float32)/(15*15)
    normalSmooth = cv2.filter2D(greenPixels, -1, kernel)
    # a kernel is basically a way to designate how to take a weighted average. The value of pixel in the kernel is the weight of that pixel. The second argument in filter2D is the desired depth. Setting to -1 means the output depth will be the same as the input. Depth means basically the dtype of numbers used. It's often 8 bits.
    cv2.imshow('normal smooth', normalSmooth)

    gausBlur = cv2.GaussianBlur(greenPixels, (15, 15), sigmaX=0, sigmaY=0)
    # basically using a gaussian kernel for filter/smoothing. Second argument is the kernel size. SigmaX and SigmaY are designate the standard deviation in x direction and y direction. Setting both to 0 it will be calculated automatically.
    cv2.imshow('gaus smooth', gausBlur)

    medianBlur = cv2.medianBlur(greenPixels, 15)
    # second argument is kernel size
    cv2.imshow('median smooth', medianBlur)

    # bilateralSmooth = cv2.bilateralFilter(greenPixels, 5, 200, 200)
    # second argument is again the size of the filter. The two last arguments indicate the smoothing strength in x and y directions. Bilateral smooth is not great with big (in size) noise
    # cv2.imshow('bilateral smooth', bilateralSmooth)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        # & 0xFF simply acquires the last 8 bits of cv2.waitKey(1)
        break
cap.release()
out.release()
cv2.destroyAllWindows()
