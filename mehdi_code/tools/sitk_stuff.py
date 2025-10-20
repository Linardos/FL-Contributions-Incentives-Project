import SimpleITK as itk



def read_nifti(image_path):
    """
    loading the data array and some of the metadata of nifti a nifti file.
    note that itk loads volumes as channel first.
    
    Parameters
    ----------
    image_path : string
        absolute path to the image file.
        
    Returns
    -------
    img_array : numpy array
        tensor array of the image data.
    img_itk : itk image
        loaded itk image.
    img_size : tuple
        image data dimension.
    img_spacing : tuple
        voxel spacing.
    img_origin : tuple
        subject coordinates.
    img_direction : tuple
        orientation of the acquired image.
    """
    
    img_itk = itk.ReadImage(image_path)
    img_size = img_itk.GetSize()
    img_spacing = img_itk.GetSpacing() 
    img_origin = img_itk.GetOrigin()
    img_direction = img_itk.GetDirection()
    img_array = itk.GetArrayFromImage(img_itk)
    
    return img_array, img_itk, img_size, img_spacing, img_origin, img_direction



def get_dicom_series(dicom_series_path):
    '''
    Get the absolute path and return dicom data+metadata

    Parameters
    ----------
    dicom_series_path : str
        absolute path to a dicom series folder.

    Returns
    -------
    img_itk : itk object
        itk image.
    img_spacing : tuple
        resolution spacing.
    img_origin : tuple
        coordinate of origin.
    img_direction : tuple
        affine matrix of direction.
    patient_tags : dictionary
        meta-data of dicom series.

    '''

    my_tags = {"patient_name": "0010|0010",
               "patient_id": "0010|0020",
               "patient_bday": "0010|0030",
               "UID": "0020|000D",
               "study_id": "0020|0010", 
               "study_date": "0008|0020",
               "study_time": "0008|0030",
               "accession_number": "0008|0050",
               "modality":"0008|0060"}
    
    series_IDs = itk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_series_path)
    series_file_names = itk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_series_path, series_IDs[0])
    series_reader = itk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    
    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    img_itk = series_reader.Execute()
    
    img_spacing = img_itk.GetSpacing()
    img_origin = img_itk.GetOrigin()
    img_direction = img_itk.GetDirection()
    
    patient_tags = {}
    for keys, vals in my_tags.items():
        if series_reader.HasMetaDataKey(0,vals):
            patient_tags[keys] = series_reader.GetMetaData(0,vals)
            
            
    return img_itk, img_spacing, img_origin, img_direction, patient_tags


def reorient_itk(itk_img):
    '''
    reorient the already loaded itk image into LPS cosine matrix.
    
    Parameters
    ----------
    itk_img : loaded itk image (not volume array)
    Returns
    -------
    reoriented files :array, itk_img, spacing, origin, direction.
    '''
    
    orientation_filter = itk.DICOMOrientImageFilter()
    orientation_filter.SetDesiredCoordinateOrientation("LPS")
    reoriented = orientation_filter.Execute(itk_img)
    reorient_array = itk.GetArrayFromImage(reoriented)
    reoriented_spacing = reoriented.GetSpacing()
    reoriented_origin = reoriented.GetOrigin()
    reoriented_direction = reoriented.GetDirection()
    
    return reorient_array, reoriented, reoriented_spacing, reoriented_origin, reoriented_direction