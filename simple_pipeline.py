# loads image, runs a tiny brightness check and writes result
import cv2
img = cv2.imread('/mnt/data/e466e5e0-c7e8-4a44-8060-06cfa2eeeec6.png')
if img is None:
    raise SystemExit("image missing")
# simple "feature": mean brightness
mean = img.mean()
print("mean brightness:", mean)
# label image with brightness value and save
cv2.putText(img, f"mean={mean:.1f}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
cv2.imwrite('/mnt/data/pipeline_out.png', img)
print("wrote /mnt/data/pipeline_out.png")
