import cv2
import numpy as np

def potential_field_with_grid(frame, model):
    grid_num = 31

    # Load class lists
    classes = ["occupied"]

    buffer_size = 1
    potential_field = np.zeros((grid_num, grid_num))

    # Object Detection
    (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.70, nmsThreshold=.7)
    obj = []

    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if len(bboxes) == 0:
        frame = cv2.putText(frame, "Land Here", (frame.shape[1]//2 - 50, frame.shape[0]//2  - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 0, 214), 2)
        frame = cv2.circle(frame, (frame.shape[1]//2 , frame.shape[0]//2 ), 5, (250, 0, 214), -1)
        return (-1, -1, frame)
    
    for class_id, score, bbox in zip(class_ids, scores, bboxes):

        (x_box, y_box, w, h) = bbox
        frame = cv2.rectangle(frame, (x_box, y_box), (x_box+w, y_box+h), (100, 110, 100), 3)
        
        x_center = (x_box + w//2)
        y_center = (y_box + h//2)

        obj.append((x_center, y_center, w, h))

        frame = cv2.circle(frame, (x_center, y_center), 5, (100, 110, 100), -1)

        for i in list(range(grid_num)):
            for j in list(range(grid_num)):
                x = i * frame.shape[1]/(grid_num - 1)
                y = j * frame.shape[0]/(grid_num - 1)
                
                try:
                    # if (x - x_center)/frame.shape[1] < 0.1 and (y - y_center)/frame.shape[0] < 0.1:
                    potential_field[j ,i] -= 0.5 * (((x - x_center)/w)**2 + ((y - y_center)/h)**2) ** 0.5
                        
                except:
                    potential_field[j ,i] += 10000


    c = np.max(np.abs(potential_field)) * 10
    # print(c)

    # print(potential_field)

    for i in range((len(range(grid_num)) - 1 )// 6):
        potential_field[i, :] += (10 - i) * c
        potential_field[-1-i, :] += (10 - i) * c
    for j in range((len(range(grid_num)) - 1) // 6):
        potential_field[:, j] += (10 - j) * c
        potential_field[:, -1-j] += (10 - j) * c
        # potential_field[i ,j] += math.sqrt((x - frame.shape[1]//2)**2 + ((y - frame.shape[0]//2)**2)) * 0.03

    for i in list(range(grid_num)):
        for j in list(range(grid_num)):
            x = i * frame.shape[1]/(grid_num - 1)
            y = j * frame.shape[0]/(grid_num - 1)

            for (x_center, y_center, w, h) in obj:

                if (x_center - w//2) <= x <= (x_center + w//2) and (y_center - h//2) <= y <= (y_center + h//2):
                    potential_field[j, i] += 10000

                if ((x - x_center)**2 + (y - y_center)**2)**0.5 < 1.3 * (w + h)/2:
                    potential_field[j, i] += 8000

    #  print(potential_field)
    
    min_index = np.unravel_index(potential_field.argmin(), potential_field.shape)
    min_y = int(min_index[0] * frame.shape[0]/(grid_num - 1))
    min_x = int(min_index[1] * frame.shape[1]/(grid_num - 1))
    

    frame = cv2.circle(frame, (min_x, min_y), 5, (250, 0, 214), -1)
    frame = cv2.putText(frame, "Land Here", (min_x - 50, min_y- 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 0, 214), 2)
    



    
    return (min_x, min_y, frame)