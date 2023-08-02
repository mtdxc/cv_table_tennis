import cv2
import numpy as np
import json
#import pickle pickle.load pickle.dump


choosen_video = 'input_video.mp4'

# Begin tracking
cap = cv2.VideoCapture(choosen_video)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print('%s (%dx%d@%d)' % (choosen_video, width, height, fps))
# trajectory parameters
t_init=0
# 1 frame represent (1/fps) second IRL then we can calculate the corresponding timestamp t at each step with:
time_step=(1/fps)
timestamp=t_init
timestamps=[]
ball_frame_positions=[]


fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
out = cv2.VideoWriter('results/tracking_videos/ball_tracking.mp4', fourcc, 25.0, (1920, 1080))

# _, frame=cap.read()
# h, w, _ = frame.shape


fgbg = cv2.createBackgroundSubtractorMOG2(
    history=15, varThreshold=50, detectShadows=False
)


# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 50
# params.maxThreshold = 100
# Filter by Color (it will be used on bw image)
params.filterByColor = True
params.blobColor = 255
# Filter by Area (remove too small and too big amount of pts)
params.filterByArea = True
params.minArea = 200
params.maxArea = 1000
# Filter by Circularity (the ball should look ~ circular)
params.filterByCircularity = True
params.minCircularity = 0.6
# Filter by Convexity (the ball should be a convex shape)
params.filterByConvexity = True
params.minConvexity = 0.5
# Filter by Inertia (degree of resemblance to a circle?)
params.filterByInertia = True
params.minInertiaRatio = 0.08

# Create a blob detector with the previous parameters
detector = cv2.SimpleBlobDetector_create(params)

frame_nb=0
limit_frame_low=390
limit_frame_up=420
nb=1
draw_trajectory = None

while True:
    
    frame_nb = frame_nb + 1
    
    # updating the time variable, taking into account the framerate of the video
    timestamp = timestamp + time_step
        
    ret, frame = cap.read()
    if frame is None:
        break  
    
    # background substraction to retain only foreground (=moving) elements, including the ball
    fg_filt=fgbg.apply(frame)
    cv2.imshow('background filtered', fg_filt)
    

    # denoising and retain only the most relevant pixel blobs with closing-opening process
    
    se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    # structuring element (mask used for filtering) which will served during the erosion step of the closing phase
    se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    # structuring element (mask used for filtering) which will served during the erosion step of the opening phase
    fg_filt = cv2.morphologyEx(fg_filt, cv2.MORPH_CLOSE, se1)
    # first we perform closing on foreground filtered bw image  
    fg_filt = cv2.morphologyEx(fg_filt, cv2.MORPH_OPEN, se2)
    # then finally opening
    cv2.imshow('after_closing_opening', fg_filt)
    

    # blob detection on preprocessed and denoised image, should capture ball pixels
    keypoints = detector.detect(fg_filt)  
    
    
    # Show current frame nb and timestamp
    cv2.putText(frame, "Frame number: {}".format(str(frame_nb)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)
    cv2.putText(frame, "Time: {}".format(str(round(timestamp,2))), (1750, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)
    
    # Show keypoints    
    if (len(keypoints)>0): 
        frame = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        for i in range(len(keypoints)):
#           cv2.circle(frame, [int(keypoints[i].pt[0]), int(keypoints[i].pt[1])], 5,(0, 255, 255), -1) 
            if (frame_nb < limit_frame_up and frame_nb > limit_frame_low):
                timestamps.append(timestamp)
                ball_x=int(keypoints[i].pt[0])
                ball_y=int(keypoints[i].pt[1])
                ball_position=[ball_x,ball_y]
                ball_frame_positions.append(ball_position)
                if draw_trajectory is None:
                    draw_trajectory = frame.copy()
                # 把球位置写到draw_trajectory
                cv2.circle(draw_trajectory, ball_position, 7,(255, 255, 255), -1)    

    cv2.imshow("Keypoints", frame)
    #out.write(frame)    

    k = cv2.waitKey(30)
    if k == ord('q'):
        break

cv2.imwrite('seq.jpg',draw_trajectory)

print('positions', ball_frame_positions)
print('timestamps', timestamps)
with open('allpt.json', 'w') as file_to_write:
    json.dump({'positions':ball_frame_positions, 'timestamps': timestamps},file_to_write)
out.release()
cap.release()
cv2.destroyAllWindows()
