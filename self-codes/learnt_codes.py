# 1. SAR
# Function to read a SAR image
# Function to convert datatype of a SAR image from float64 to uint8
# Code to read a label image and generate a color map
# Code to make a video from a time-series of SAR images
# 2. ML/DL
# Function to save a deep-learning model in .h5 format
# Function to compute accuracy in binary classification
# Function to plot a confusion matrix
# Function to plot model loss of several splits (n_splits) when using cross-validation
# Function to plot model accuracy of several splits (n_splits) when using cross-validation
# Function to create a master confusion matrix from matrices of all splits when using cross-validation
# Function to create image generator with multiple inputs
# Code to divide training data into chunks
# Methods to save large matrices in disc
# Function to load data from .h5 file sequentially or randomly

# 3. Image Processing
# Function to extract a patch centered as (i,j) of size 2d+1 from image im
# 4. Python
# Function to create custom color map with matplotlib
# 5. System
# Code to get environement setting

# 1. SAR
# Function to read a SAR image

def readFile(file):
    (fileroot, fileExt) = os.path.splitext(file)
    ds = gdal.Open(file)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    initial_shape = arr.shape
    arr = makeUint8Image(arr)
    return arr
#Function to convert datatype of a SAR image from float64 to uint8

    def makeUint8Image(arr_in):
        arr = arr_in.flatten()
        x = pd.Series(arr)
        y = np.nan_to_num(arr_in)
        x = x.dropna()
        vmin = np.percentile(x,  1)
        vmax = np.percentile(x, 99)
        u = (y-vmin)/(vmax-vmin) #scaling in the range 0-1
        u[u>1.0] = 1.0
        u[u<0.0] = 0.0
        v = 255*(u**0.4545)
        return v.astype('uint8')
#Code to read a label image and generate a color map

    labelfilepath = "/home/saror/work/shreya/NNGPU/RG_project/data/labels/odaiba_gt.tiff"

    dataset = gdal.Open(labelfilepath, gdalconst.GA_ReadOnly)
    label = dataset.GetRasterBand(1).ReadAsArray()
    label = cv2.resize(label, dsize=( column, row), interpolation=cv2.INTER_LINEAR)
    colortable = dataset.GetRasterBand(1).GetRasterColorTable() #NULL

    tmpimg = label.astype(np.float)/label.max().astype(np.float)
    tmpimg2 = (cm.jet(tmpimg)*255).astype(np.uint8)
#Code to make a video from a time-series of SAR images

    #variables
    image_folder= "/mnt/hdd/shreya/parkingLot2"
    video_name = "00video.avi"
    fps=1

    output_folder= "/home/saror/work/shreya/NNGPU/change_detection"
    #UX,UY= 130.888663657, 33.9943643658  #upper right point #Obtained form QGIS
    #LX,LY= 130.906475072, 33.9821549276  #lower right point

    files= os.listdir(image_folder)
    images = [img for img in files if img.endswith(".tif")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width = frame.shape

    fourcc= cv2.VideoWriter_fourcc(*'DIB ')
    video= cv2.VideoWriter(video_name, fourcc, fps, (width, height))
    for file in files:
        if file.endswith('tif'):  
            #os.system("gdal_translate -projwin "+ str(UX) +" "+ str(UY)+" " + str(LX)+ " " +str(LY) + " -of GTiff "+ image_folder + '/'+file+" "+ output_folder +'/'+ file )
            (fileroot, fileExt)= os.path.splitext(file)
            #datetime = fileroot[14:22]
            ds= gdal.Open(image_folder+'/' +file)
            arr= ds.GetRasterBand(1).ReadAsArray()
            v = makeUint8Image(arr)
            v = cv2.merge([v,v,v])
            #v = cv2.putText(v, datetime, (0,30), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255),2)
            video.write(v)

    del images
    cv2.destroyAllWindows()
    video.release()
#2. ML/DL
#Function to save a keras model in .h5 format

     def saveModel(model, location):
        model_json = model.to_json()
        with open(location, 'w') as json_file: 
            json_file.write(model_json)
        sparse_autoencoder.save_weights('model_weights.h5')
#Function to compute accuracy in binary classification

    def compute_accuracy(y_true, y_pred):

         pred = y_pred.ravel() > 0.5
         return np.mean(pred == y_true)
#Function to plot a confusion matrix

    def plot_confusion_matrix(cm, classes, normalize = False, title = 'confusion matrix', cmap = plt.cm.Reds):
        if normalize:
            cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix") 
        else:
            print("Confusion matrix without normalization")
        print(cm)
        plt.imshow(cm, interpolation = 'nearest', cmap= cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(tick_marks, classes, rotation = 45)
        plt.yticks(tick_marks, classes)
        plt.tight_layout()

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        return plt
#Function to plot model loss of several splits (n_splits) when using cross-validation

     def plot_loss(train_loss, val_loss, n_splits, n_epochs):  
        train_loss_mat= np.zeros((n_splits, n_epochs))
        val_loss_mat= np.zeros((n_splits, n_epochs))
        for i in range(0, len(train_loss)):
            train_loss_mat[i,:]= train_loss[i]
            plt.plot(np.array(train_loss[i]).T,color= 'black')
        mean_loss= np.mean(train_loss_mat, axis=0)    
        plt.plot(mean_loss.T, linewidth= 3.0, color= 'black', label = 'train_loss')
        for i in range(0, len(val_loss)):
            val_loss_mat[i,:]= val_loss[i]
            plt.plot(np.array(val_loss[i]).T,color= 'red')
        mean_val_loss= np.mean(val_loss_mat, axis=0)    
        plt.plot(mean_val_loss.T, linewidth= 3.0, color= 'red', label= 'val_loss')
        plt.legend(loc= 'upper right')
        plt.show()
#Function to plot model accuracy of several splits (n_splits) when using cross-validation

    def plot_acc(train_acc, val_acc, n_splits, n_epochs):    #when using CV
        train_acc_mat= np.zeros((n_splits, n_epochs))
        val_acc_mat= np.zeros((n_splits, n_epochs))
        for i in range(0, len(train_acc)):
            train_acc_mat[i,:]= train_acc[i]
            plt.plot(np.array(train_acc[i]).T, color= 'black')
        mean_acc= np.mean(train_acc_mat, axis=0)    
        plt.plot(mean_acc.T, linewidth= 3.0, color= 'black', label = 'train_acc')
        for i in range(0, len(val_acc)):
            val_acc_mat[i,:]= val_acc[i]
            plt.plot(np.array(val_acc[i]).T, color= 'red')
        mean_val_acc= np.mean(val_acc_mat, axis=0)    
        plt.plot(mean_val_acc.T, linewidth= 3.0, color= 'red', label= 'val_acc')
        plt.legend(loc= 'upper right')
        plt.show()
#Function to create a master confusion matrix from matrices of all splits when using cross-validation

    def create_conf_mat(n_splits, n_classes, conf_mats):
        master_conf_mat=np.zeros((n_splits,n_classes**2))
        for i in range(0, len(conf_mats)):
            c= conf_mats[i].ravel()
            master_conf_mat[i]= c
        master_conf_mat=np.sum(master_conf_mat, axis=0).reshape(4,4)
        return master_conf_mat
#Function to create image generator with multiple inputs
#REF: https://www.kaggle.com/sinkie/keras-data-augmentation-with-multiple-inputs

    def gen_flow_for_two_inputs(xtrain, sel_angletrain, ytrain, seed_value):
        genX1= datagen.flow(xtrain, ytrain, batch_size=batch_size, seed= seed_value)
        genX2= datagen.flow(xtrain, sel_angletrain, batch_size= batch_size, seed= seed_value)
        while True:
            xtraini= genX1.next()
            sel_angletraini= genX2.next()
            yield [xtraini[0], sel_angletraini[1]], xtraini[1]
#Code to divide training data into chunks

    chunk_size= 3000 #choose appropriate chunk size
    chunk_start=0
    chunk_end = chunk_start + chunk_size

    if (len(train_samples)%chunk_size)==0:
        n_chunks=len(train_samples)//chunk_size
    else:
        n_chunks= len(train_samples)//chunk_size+1

    for chunk in range (0, n_chunks):
        train_features = []
        train_class = []
        for x in range (chunk_start, chunk_end):

            if (x<len(train_samples)):
               for sample in train_samples:
                   feature = #extract training feature
                   clas = #extract training class
               train_features.append(feature)
               train_class.append(clas)

        train_features= np.array(train_features)%
        train_class= np.array(train_class)

        h5f = h5py.File('/path_to_folder/fileName_chunk_%d.h5' % chunk, 'w')
        h5f.create_dataset('train_features', data = train_features)
        h5f.create_dataset('train_class', data = train_class)
        h5f.close()
        print(chunk_start, chunk_end)%

        chunk_start = chunk_start + chunk_size
        chunk_end = chunk_start + chunk_size

    print(time.time()-start_time)
#Methods to save large matrices in disc
#REF: https://stackoverflow.com/questions/9619199/best-way-to-preserve-numpy-arrays-on-disk

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.33, random_state = 42, stratify = y)

    #Method 1: Using hDF5 format (can store many data types, fast to read and write but can corrupt easily)

    #write
    h5f = h5py.File('path_to_file/file_train.h5', 'w')
    h5f.create_dataset('X_train', data = X_train)
    h5f.create_dataset('y_train', data = y_train)
    h5f.close()

    #read
    h5f = h5py.File('path_to_file/file_train.h5', 'r')
    train_features = h5f['X_train'][:]
    train_class = h5f['y_train'][:]
    h5f.close()

    #Method 2: using npz (save multple arrays in .npy format, specificaly designed for numpy arrays)
    outfile = 'path_to_file/file_train.npz'
    np.savez(outfile, X_train= X_train, y_train=y_train, X_val=X_val, y_val= y_val)

    readArrays = np.load(outfile)
    print(readArrays.files)

    #Method 3: Pickle (serializes python objects to store them on disk, can not be across different languages)
    #The python objects can contain different data types, booleans, Integers, Floats, tuples etc.
    data = {'X_train':X_train, 'y_train': y_train}
    outfile = open('file_train.pickle', 'wb')
    pickle.dump(data, outfile)
    outfile.close()

    outfile = open('file_train.pickle','rb')
    data = pickle.load(outfile)
    outfile.close()

    #Method 4: Hickle (HDF5 version of pickle), usage same as pickle

#Function to load data from .h5 file

    #h5f file should have 4 keys ['train_features','train_class','val_features','validation_class']

    h5f = h5py.File('/mnt/hdd/shreya/change_detection_files/san_test_data.h5', 'r')

    #to read training samples sequentially in size = batch_size 
    def imageLoader1(h5f, batch_size):
        L= len(h5f['train_features']) #number of training images
        while True:
            batch_start=0
            batch_end= batch_size
            while batch_start<L:
                limit = min(batch_end, L)
                X= h5f['train_features'][batch_start:limit]
                Y= h5f['train_class'][batch_start:limit]
                yield (X,keras.utils.to_categorical(Y,2))     
                print(batch_start, batch_end)

                batch_start +=batch_size
                batch_end +=batch_size

    #to read training samples randomly in size = batch_size

    def imageLoader2(features, labels, batch_size):
    # Create empty arrays to contain batch of features and labels
     batch_features = np.zeros((batch_size, 64, 64, 3))
     batch_labels = np.zeros((batch_size,1))
     while True:
       for i in range(batch_size):
         # choose random index in features
         index= random.choice(len(features),1)
         batch_features[i] = h5f['train_features'][index]
         batch_labels[i] = h5f['train_class'][index]
       yield batch_features, batch_labels
#3. Image Processing
#Function to extract a patch centered as (i,j) of size 2d+1 from image im

    def neighbors(im, i, j, d=1):
        n = im[i-d:i+d+1, j-d:j+d+1].flatten()
        return n
#4. Python
#Function to create custom color map with matplotlib

    def generate_cmap(colors):
        values = range(len(colors))
        vmax = np.ceil(np.max(values))
        color_list = []
        for v, c in zip(values, colors):
            color_list.append( ( v/ vmax, c) )
        return LinearSegmentedColormap.from_list('custom_cmap', color_list)        


    #Usage 1:   

    cm  =  generate_cmap ([ 'moccasin', 'black', 'white' ]) #one color is assigned to one value in array
    plt.imshow(array, cmap = cm)
    plt.show() 

    #Usage 2: to create an image highlighting FP, FN, TP and TN

    def assign_value_for_cmap(y_actual, y_pred):
        row, column = initial_shape
        color_map = np.zeros((row, column))
        for i in range(len(y_pred)): 
            x = locations[i][0]
            y = locations[i][1]
            if y_actual[i]==y_pred[i]==0:
               color_map[x,y]= 1
            if y_actual[i]==y_pred[i]==1:
               color_map[x,y]= 2
            if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
               color_map[x,y]= 3
            if y_pred[i]==0 and y_actual[i]!=y_pred[i]:
               color_map[x,y]= 4
            return color_map

    color_map = assign_value_for_cmap(y_actual, y_pred, locations) 
    cm  =  generate_cmap ([ 'moccasin', 'black', 'white', 'green', 'red' ]) #one color is assigned to one value
    plt.imshow(color_map, cmap = cm)
    plt.show()
#5. System
#Code to get environement setting

    import IPython
    # print system information (but not packages)
    print(IPython.sys_info())

    # get module information
    !pip freeze > frozen-requirements.txt

    # append system information to file
    with open("frozen-requirements.txt", "a") as file:
        file.write(IPython.sys_info())