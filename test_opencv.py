import cv2
img = cv2.imread('/mnt/f/iit/test.jpg')
if img is None:
    print("ERROR: image not found")
else:
    h,w = img.shape[:2]
    print("Loaded image size:", w, "x", h)
    # draw a small rectangle and save as output
    cv2.rectangle(img, (10,10), (w-10,h-10), (0,255,0), 3)
    out = '/mnt/data/opencv_test_out.png'
    cv2.imwrite(out, img)
    print("Wrote output to", out)
