def draw_flow(img1, kp1, kp2):
    """Draws a line from kp1 to kp2 to img1."""

    import cv2
    import numpy as np
    import os

    outimg = np.array(img1)
    kpoints = np.hstack((kp1, kp2))
    for kp in kpoints:
        cv2.line(
            outimg,
            (int(kp[0]), int(kp[1])),
            (int(kp[2]), int(kp[3])),
            color=(255, 0, 0),
        )
        cv2.circle(outimg, (int(kp[0]), int(kp[1])), 10, (0, 0, 155), 3)
        cv2.circle(outimg, (int(kp[2]), int(kp[3])), 10, (0, 0, 255), 3)

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", outimg)
    cv2.waitKey()
