import numpy as np 
import cv2 
import os
import shutil 
import matplotlib.pyplot as plt 

def horizontal_projection(image, threshold=5):
    '''
        project all pixels horizontally
        
        Parameters:
            image:  m x n 
            threshold: Number
                       The threshold of minimum pixel to be project. Default is 5
        
        return:
            projection: list of Number
                
    '''
    image = cv2.adaptiveThreshold(image, 1 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)
    
    projection = np.sum(image, 1) 
    projection = np.where(projection > threshold, projection, 0 )
    
    return projection

######################################################################

def vertical_projection(image, threshold=5):
    '''
        project all pixels vertically
        
        Parameters:
            image:  m x n 
            threshold: Number
                       The threshold of minimum pixel to be project. Default is 5
        
        return:
            projection: list of Number
                
    '''
    image = cv2.adaptiveThreshold(image, 1 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)

    projection = np.sum(image, 0)
    projection = np.where(projection > threshold, projection, 0)
    
    return projection

######################################################################



def plot_horizontal_projection(document, horizontal_projection, w=1000):
    '''
    '''
    
    m = np.max(horizontal_projection)
    result = np.zeros((horizontal_projection.shape[0],w))


    # Draw a line for each row
    for row in range(document.shape[0]):
        cv2.line(result, (0,row), (int(horizontal_projection[row]*w/m),row), (255,255,255), 1)
        
    cv2.imshow('Original', document)    
    cv2.imshow('Horizontal Projection', result)
    cv2.waitKey(0)

#     plt.plot(y=horizontal_projection)
#     plt.ylim([document.shape[0], 0])
#     plt.show()

######################################################################

def plot_vertical_projection(document, vertical_projection, w=1000, enlarge = 1 ):
    '''
    '''
    
    m = np.max(horizontal_projection)
    result = np.zeros((document.shape[0] * enlarge, document.shape[1]))


    # Draw a line for each row
    for row in range(document.shape[1]):
        cv2.line(result, (row, 0), (row, int(horizontal_projection[row]*w/m)), (255,255,255), 1)
        
    cv2.imshow('Original', document)    
    cv2.imshow('Vertical Projection', result)
    cv2.waitKey(0)

######################################################################
        
def segment_line(line, vertical_projection, target_path):
    '''
        Segment the line into characters and save.
        
        Args:
            image: (nxm)
                the image to be segmented
            vertical_projection: (mx1)
                a list of projection value of the image
            file_path: string
                the path of result to be saved
                
    '''
    start = []
    end = []
    
    is_start = True
    selected = False 
    first = 0
    second = 0
    
    for index, column in enumerate(vertical_projection):
        if column == 0 and not selected:
            first = index 
            selected = True

        elif selected and column != 0 :
            second = index - 1 
            mid = int( (first + second)/2 )
            selected = False
            
            
            if is_start:
                start.append(mid)
                is_start = False
            else:
                end.append(mid)
                is_start = True
    
    
    try:
        os.makedirs(target_path)
    except FileExistsError as e: 
        shutil.rmtree(target_path, ignore_errors = True)
        os.makedirs(target_path, exist_ok = True)
    except OSError as e:  
        if e.errno != os.errno.EEXIST:
            raise   
            
    for index, (start, end) in enumerate(zip(start, end)):
        cv2.imwrite(f'{target_path}\\{index}.png', line[:, start:end +1 ])
        
######################################################################



def segment_document(document, horizontal_projection, target_path):
    '''
        Segment the document into lines using horizontal projection.
        
        Args:
            document: (nxm)
                the document image to be segmented
            horizontal_projection: (nx1)
                the horizontal projection value of the document 
            path: string
                the path of the segmented lines to be save
            
    '''
    line_start = []
    line_end = []

    is_start = True
    selected = False 
    first = 0
    second = 0

    for index, line in enumerate(horizontal_projection):
        if line == 0 and not selected:
            first = index
            selected = True
        elif selected and line !=0 :
            second = index - 1 
            mid = int((first + second) /2 )
            selected = False 

            if is_start: 
                line_start.append(mid)
                is_start = False 
            else:
                line_end.append(mid)
                is_start = True
    
   
    try:
        os.makedirs(target_path)
    except FileExistsError as e: 
        shutil.rmtree(target_path, ignore_errors = True)
        os.makedirs(target_path, exist_ok = True)
    except OSError as e:  
        if e.errno != os.errno.EEXIST:
            raise   
            
    for index, (start, end) in enumerate(zip(line_start, line_end)):
        cv2.imwrite(f'{target_path}\\Line{index}.png', document[start:end, :])
        
######################################################################

def segment_document_and_save(document, horizontal_projection, target_path):
    line_start = []
    line_end = []

    is_start = True
    selected = False 
    is_append = False 
    first = 0
    second = 0

    min_gap = 3
    
    current_line = 0 #line number starts from 0 

    i = 0
    
    for index, line in enumerate(horizontal_projection):
        
        if not selected and line != 0:
            gap = np.nan
            if not is_start: 
                gap = index - line_end[-1]

            if gap < min_gap:     
                is_append = True
                selected = True
            else:    
                line_start.append( index - 1 )
                current_line += 1 
                selected = True
            is_start = False 
            
        elif selected and line == 0: 
            if not is_append: 
                line_end.append(index)
                selected = False 
            else: 
                line_end[-1] = index 
                selected = False
                is_append = False
             
            
   
            
    try:
        os.makedirs(target_path)
    except FileExistsError as e: 
        shutil.rmtree(target_path, ignore_errors = True)
        os.makedirs(target_path, exist_ok = True)
    except OSError as e:  
        if e.errno != os.errno.EEXIST:
            raise   
    

    if len(line_start) > 0 and len(line_end) > 0 : 
        line_start.pop(0)
        line_end.pop(0)
        
    for index, (start, end) in enumerate(zip(line_start, line_end)):
        cv2.imwrite(f'{target_path}\\Line {index}.png', document[start:end, :])
                


