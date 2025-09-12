import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
from osgeo import gdal
import exiftool
from tensorflow.keras.models import load_model
import pickle
import multiprocessing
import warnings
warnings.filterwarnings("ignore")


# import rasterio as rio
# from rasterio.warp import transform 
# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.preprocessing.image import load_img


import fonksiyonlar as fk    #fonksiyonların olduğu dosya çağrılır

def match(img,template):
    methods =['cv2.TM_CCOEFF']
    method  = eval(methods[0])
    
    res= cv2.matchTemplate(img, template, method, None)
    return res
    





dirname = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':

  

    
    pool = multiprocessing.Pool()
    pool = multiprocessing.Pool(processes=4)
    
    
    
    #haritalar klasöründeki ilk görüntüde DEM verileri vardır. ikinci görüntü ise normal rgb görüntüdür.
    harita_yol=dirname+'/haritalar/'
    harita_yol_list=os.listdir(harita_yol)
    model_yol=dirname+'/model/'
    model_list=os.listdir(model_yol)
    #ana_harita_elevation="urgup_gmap_30_cm_elevations_560.tif"
    ana_harita_elevation="ana_harita_karlik_30_cm_bingmap_elevations_576.tif"    
    
    
    
    
    # haritadaki piksellerin gps koordinatları bulunur ve koordinatlar olarak ayrı bri dosya olarak diske kaydedilir. bir kez çalıştırılması yeterlidir
    ###############################################################################
    """
    import rasterio
    from affine import Affine
    from pyproj import Proj, transform
    
    #fname = 'urgup_gmap_georef.tif'
    fname = ana_harita
    
    # Read raster
    with rasterio.open(fname) as r:
        T0 = r.transform  # upper-left pixel corner affine transform
        p1 = Proj(r.crs)
        A = r.read()  # pixel values
    
    # All rows and columns
    cols, rows = np.meshgrid(np.arange(A.shape[2]), np.arange(A.shape[1]))
    
    # Get affine transform for pixel centres
    T1 = T0 * Affine.translation(0.5, 0.5)
    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    rc2en = lambda r, c: (c, r) * T1
    
    # All eastings and northings (there is probably a faster way to do this)
    eastings, northings = np.vectorize(rc2en, otypes=[float, float])(rows, cols)
    
    # Project all longitudes, latitudes
    p2 = Proj(proj='latlong',datum='WGS84')
    longs, lats = transform(p1, p2, eastings, northings)
    
    
    coordinates = (longs,lats)
    
    
    
    
    pickle_out = open("koordinatlar.pickle","wb")
    pickle.dump(coordinates, pickle_out)
    pickle_out.close()
    """
    
    
    
    import rasterio
    from affine import Affine
    from pyproj import Proj, transform
    
    #fname = 'urgup_gmap_georef.tif'
    fname = harita_yol+harita_yol_list[0]
    
    
    # Read raster
    with rasterio.open(fname) as r:
        T0 = r.transform  # upper-left pixel corner affine transform
        p1 = Proj(r.crs)
        A = r.read()  # pixel values
    
    # All rows and columns
    cols, rows = np.meshgrid(np.arange(A.shape[2]), np.arange(A.shape[1]))
    
    def koordinat_bul(row,col):
        # Get affine transform for pixel centres
        T1 = T0 * Affine.translation(0.5, 0.5)
        # Function to convert pixel row/column index (from 0) to easting/northing at centre
        rc2en = lambda r, c: (c, r) * T1
        
        # All eastings and northings (there is probably a faster way to do this)
        eastings, northings = np.vectorize(rc2en, otypes=[float, float])(rows[row], cols[col])
        
        
        # Project all longitudes, latitudes
        p2 = Proj(proj='latlong',datum='WGS84')
        longs, lats = transform(p1, p2, eastings, northings)
       
        
        return (longs,lats)
    
    
    
    
    
    
    ###############################################################################
    
    #DEM verileri aktarılır
    
    filename = ana_harita_elevation
            
    dataset = gdal.Open(filename)
    
    gt = dataset.GetGeoTransform()
    band = dataset.GetRasterBand(1)  #5. bant elevation bandı
    
    DEM_array = band.ReadAsArray()    
    
    ###############################################################################
    
    #%%
    
    
    
    sonuclar = []
    
    
    for k in range(len(harita_yol_list)):
        konum=(0,0)
        kare=()    
        dogru_tahmin=0
        yanlis_tahmin=0
        ana_harita="haritalar/"+harita_yol_list[k]
          
        img = cv2.imread(ana_harita,0) #haritalar klasöründeki ikinci görüntüyü okur
        print(img.shape)
          
        kenarx=int(img.shape[0]/512)
        
        #parcalar klasöründeki anlık görüntüleri getirir
        anlik_yol=dirname+'/parcalar/'
        anlik_yol_list=os.listdir(anlik_yol)
        
        #anlik_goruntu=anlik_yol+anlik_yol_list[0]
        
        anlik_yol_list = sorted( anlik_yol_list,
                                key = lambda x: os.path.getmtime(os.path.join(anlik_yol, x))  # tarihe göre klasördeki dosyaları sıralar
                                )
     
        
        
        img_temp=cv2.imread(ana_harita) 
        for i in range(len(anlik_yol_list)):
            
            img = cv2.imread(ana_harita,0)
            print(img.shape)
            anlik_goruntu = "parcalar/"+anlik_yol_list[i]  #klasördeki ilk görüntüyü getir
        
            #exif bilgileri okunur
            #####################################################
            with exiftool.ExifToolHelper() as et:
                metadata = et.get_metadata(anlik_goruntu)
            
                
                #gimbal_yaw değerini getirir. 
            yaw =float(metadata[0]["XMP:FlightYawDegree"])
            #yaw =float(metadata[0]["XMP:GimbalYawDegree"])
            
            altitude=metadata[0]["EXIF:GPSAltitude"]
            
            gps_latitude = metadata[0]["EXIF:GPSLatitude"]
            
            gps_longitude =metadata[0]["EXIF:GPSLongitude"]
            ######################################################
            
            #anlık görüntünün ana haritada karşılık geldiği rakım değeri bulunur
            knm=fk.piksel_bul(ana_harita,gps_longitude, gps_latitude)
            if knm[0]<256 or knm[0]>img.shape[0]-256:
                print("dışarıda")
                continue
            elif knm[1]<256 or knm[1]>img.shape[1]-256:
                print("dışarıda")
                continue
            
            rakim=DEM_array[knm[1],knm[0]]
            """
            try:
                rakim=DEM_array[knm[1],knm[0]]
               
            except:
                print("dışarıda")
                continue
            """
            
            #spatial çözünürlük elde etme
            if metadata[0]["EXIF:Model"]=="L1D-20c":   
                #spatial çözünürlük elde etme
                #######################################################################
                camera_sensor_genislik=15.9 #mavic2pro için 13.2  milimetre sensör genişliği
                camera_focal_lenght=metadata[0]["EXIF:FocalLength"] #mavic2pro için 10.26 milimetre
                ucus_yuksekligi=altitude - rakim + 29 #metre olarak yerden x"x""uçuş yüksekliği  35 dem dosyasındaki hatadan dolayı
                goruntu_piksel_genisligi = 5472 #pipksel olarak resmin genişliği
                goruntu_piksel_yuksekligi = 3648 #pipksel olarak resmin genişliği
                mekansal_cozunurluk = (camera_sensor_genislik*ucus_yuksekligi*100)/(camera_focal_lenght*goruntu_piksel_genisligi)  #mekansal çözünürlük cantimeter/pixel olarak
                goruntunun_gercek_uzunlugu=(mekansal_cozunurluk*goruntu_piksel_genisligi)/100 #metre olarak
                
                #görüntünün hangi oranda küçültüleceğini belirler mekansal çözünürlüğe göre
                #olcek_scale_test=(mekansal_cozunurluk/(29.9 *(560/544)))
                
                #olcek_scale_test=(mekansal_cozunurluk/29.85 ) * (560/544)
                olcek_scale_test=(mekansal_cozunurluk/29.85 ) * (576/544)
                
           
                
              
              
                
                #######################################################################
            
            
            elif metadata[0]["EXIF:Model"]=="FC2204":   
                #spatial çözünürlük elde etme
                #######################################################################
                camera_sensor_genislik =  8.407036405 #mavic2zoom için 6.17  milimetre sensör genişliği
                camera_focal_lenght= metadata[0]["EXIF:FocalLength"]  #mavic2zoom için 4 milimetre
                ucus_yuksekligi=altitude - rakim   #metre olarak yerden x"x""uçuş yüksekliği  33 dem dosyasındaki hatadan dolayı
                #←ucus_yuksekligi=726
                goruntu_piksel_genisligi = 4000 #pipksel olarak resmin genişliği
                goruntu_piksel_yuksekligi = 3000 #pipksel olarak resmin genişliği
                mekansal_cozunurluk = (camera_sensor_genislik*ucus_yuksekligi*100)/(camera_focal_lenght*goruntu_piksel_genisligi)  #mekansal çözünürlük cantimeter/pixel olarak
                goruntunun_gercek_uzunlugu=(mekansal_cozunurluk*goruntu_piksel_genisligi)/100 #metre olarak
               
                #görüntünün hangi oranda küçültüleceğini belirler mekansal çözünürlüğe göre
                #olcek_scale_test=(mekansal_cozunurluk/29.85)  * (560/544)
                olcek_scale_test=(mekansal_cozunurluk/29.85 ) * (576/544)
                #######################################################################
            print("\n")
            #print(olcek_scale)
            print(olcek_scale_test)
            print("\n")
            
           
            
            
            
          
            
            #################################################################################################
            
            # Reading the image
            image = cv2.imread(anlik_goruntu)
            
            # dim=(1000,750)
            
            # image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            
            # dividing height and width by 2 to get the center of the image
            height, width = image.shape[:2]
            # #get the center coordinates of the image to create the 2D rotation matrix
            #center = (int(width/2), int(height/2))
            
            # #using cv2.getRotationMatrix2D() to get the rotation matrix
            # #scale parametresi ile görüntünün spartial çözünürlüğü 60 cm'ye ayarlanır
            # #angle ile görüntünün yav değerinin tam tersine rotate edilir ve görüntü kuzeye döndürülür.
            # rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=(-1*yaw), scale=olcek_scale)
            
            
            # #rotate the image using cv2.warpAffine
            # rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height), borderValue=(255,255,255))
            
            
            angle=-yaw
            rimage = fk.rotate_image(image, angle)
            #cv2.imshow("rotated",rimage)
            
            t=fk.rotated_rect(width,height, angle)
                
            cr_image = fk.crop_around_center(rimage,int(t[0]), int(t[1]))
            
            height,width= (cr_image.shape[0],cr_image.shape[1])
            
            rotated_image=cv2.resize(cr_image, (int(width*olcek_scale_test),int(height*olcek_scale_test)),interpolation=cv2.INTER_NEAREST ) 
            
            
            #çözünürlüğü 60 cm'ye ayarlanmış görüntünün orta noktası bulnur
            height, width = rotated_image.shape[:2]
            # get the center coordinates of the image to create the 2D rotation matrix
            center = (int(width/2), int(height/2))
            
            fark=np.minimum(center[0],center[1])-288    # 576'lık frame'in elde edilen dikdörtgenin dışına taşmaması için yazıldı 
            if fark>100:
                fark=100
            
            rotated_part1 = rotated_image[center[1]-288-fark:center[1]+288-fark,center[0]-288-fark:center[0]+288-fark]
            rotated_part2 = rotated_image[center[1]-288:center[1]+288,center[0]-288:center[0]+288]
            rotated_part3 = rotated_image[center[1]-288+fark:center[1]+288+fark,center[0]-288+fark:center[0]+288+fark]
            
            
            cv2.namedWindow('Rotated part', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Rotated part', 400, 400)
            

            cv2.imshow('Rotated part', rotated_part2)
            _ = cv2.waitKey(1) 
            
            rotated_part1 = cv2.cvtColor(rotated_part1, cv2.COLOR_BGR2GRAY)
            rotated_part2 = cv2.cvtColor(rotated_part2, cv2.COLOR_BGR2GRAY)
            rotated_part3 = cv2.cvtColor(rotated_part3, cv2.COLOR_BGR2GRAY)
            
            template=[]
            
            template.append(rotated_part1)
            template.append(rotated_part2)
            template.append(rotated_part3)
            ######################################################################################################
            
            # template = anlik_goruntu
            # template = cv2.imread(template,0)
            # plt.imshow(template, cmap = "gray")
            
            for j in range(3):
                print(template[j].shape)
                h,w =template[j].shape
                
                template_resized = cv2.resize(template[j], (544,544),interpolation=cv2.INTER_NEAREST ) 
                template_resized = ((template_resized - 127.5) / 127.5)   #template_resized2 = template_resized1.astype('float32')/255.0
                    
                template_resized=template_resized.reshape(-1,544,544,1)    
                      
                
                
                model_yolu=model_yol+model_list[k]
                
                model = load_model(model_yolu)
                template[j] = model.predict(template_resized)
                
                template[j]=template[j].reshape(544,544)
                
                template[j] = ((template[j]*127.5)+127.5)  
                 
                  
                template[j] = template[j].astype('uint8') 
                
                template[j]=template[j][16:528,16:528]
                
                #plt.imshow(template[j], cmap = "gray")
                
            cv2.namedWindow('model uygulanmis', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('model uygulanmis', 400, 400)
            cv2.imshow("model uygulanmis", template[1])
            _ = cv2.waitKey(1) 
                
            print((i+1),"/",(len(anlik_yol_list)))
            
            
            
            
            
            
            # gdal.Warp('anlik_goruntu_warped.tif', anlik_goruntu, xRes=0.09, yRes=0.09) 
            # raster = gdal.Open('anlik_goruntu_warped.tif')
            # gt =raster.GetGeoTransform()
            
            # print (gt)
            # pixelSizeX = gt[1]
            # pixelSizeY = -gt[5]
            # print ("x = ",pixelSizeX)
            # print ("y = ",pixelSizeY)
            
            
            
            
            
            
            #methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            #           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
            
            
            #paralel programlama ile aynı anda 3 templatematching yapılır
            inputs=[(img,template[0]),(img,template[1]),(img,template[2])]
            outputs1 = [pool.apply_async(match, args=inpt) for inpt in inputs]
            
                      
            res1=outputs1[0].get()
            res2=outputs1[1].get()
            res3=outputs1[2].get()
            #methods =['cv2.TM_CCOEFF']
            #for meth in methods:
                #method  = eval(meth)    #stringleri fonksiyona çeviren fonksiyona
                # res1= cv2.matchTemplate(img, template[0], method, None)
                # res2= cv2.matchTemplate(img, template[1], method, None)
                # res3= cv2.matchTemplate(img, template[2], method, None)
            print(res1.shape)
            min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
            min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
            min_val2, max_val3, min_loc3, max_loc3 = cv2.minMaxLoc(res3)
                
            print(min_val2, max_val2, min_loc2, max_loc2)
                
                # if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
                #     top_left1 =min_loc1
                # else:
            top_left1 = max_loc1
            top_left2 = max_loc2
            top_left3 = max_loc3
                        
            bottom_right1 = (top_left1[0] + w,top_left1[1] + h)
            bottom_right2 = (top_left2[0] + w,top_left2[1] + h)
            bottom_right3 = (top_left3[0] + w,top_left3[1] + h)
            
                #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                
            global_x = max_loc2[0]+int(w/2)
            global_y = max_loc2[1]+int(h/2)
                 
                 
                 
            a=(top_left1[0],top_left1[1],w,h)
            b=(top_left2[0],top_left2[1],w,h)
            c=(top_left3[0],top_left3[1],w,h)
                 
                 
                 #konum bulmak için kesişimler ve kesişim karelerinin koordinatları bulunuyor
            kesisim_ab = fk.intersection(a, b);
            kesisim_bc = fk.intersection(b, c);
            kesisim_ac = fk.intersection(a, c);
                
                
            if kesisim_ab!=() and kesisim_bc!=() and kesisim_ac!=():
                kesisim_abc=fk.intersection(kesisim_ab, kesisim_bc)
                kare=(kesisim_abc[0],kesisim_abc[1],int(kesisim_abc[2]),int(kesisim_abc[3]))
                print("konum: ",kare)
            elif kesisim_ab!=() :
                kare=(kesisim_ab[0],kesisim_ab[1],int(kesisim_ab[2]),int(kesisim_ab[3]))
                print("konum: ",kare)
            elif kesisim_bc!=() :
                kare=(kesisim_bc[0],kesisim_bc[1],int(kesisim_bc[2]),int(kesisim_bc[3]))
                print("konum: ",kare) 
            elif kesisim_ac!=() :
                kare=(kesisim_ac[0],kesisim_ac[1],int(kesisim_ac[2]),int(kesisim_ac[3]))
                print("konum: ",kare) 
                        
            else:
                print("kesişim yok")
                kare=(0,0,0,0)
                kare=b
                
            konum = (kare[0]+int(kare[2]/2),kare[1]+int(kare[3]/2))
            
            """
                gps_longtidye ce gps_latitde değişkenleri anlık görüntünün korrdinatlarını verir
                koordinatlar[1][konum[0]][konum[1]] ise modelin tahmin ettiği konumun koordinatlarını verir
                ve aralarındaki uzaklık hesaplanır.    
            """
            
            koordinatlar=koordinat_bul((konum[1],konum[0]),(konum[1],konum[0]))

            lat_tahmin = koordinatlar[1]
            long_tahmin = koordinatlar[0]
                
            uzaklik = fk.quick_distance(gps_latitude,gps_longitude,lat_tahmin,long_tahmin)    
            print("uzaklik = {:.2f} km".format(uzaklik)) 
                
            
            
            centerOfCircle=(knm[1],knm[0])
            radius=10
            if(uzaklik<0.15):
                dogru_tahmin+=1
                sonuclar.append([[anlik_yol_list[i]],["Dogru"],[(gps_latitude,gps_longitude)],[(lat_tahmin,long_tahmin)]])
                cv2.circle(img_temp, centerOfCircle, radius, (0,255,0), 25) 
                    
            else:
                yanlis_tahmin+=1
                sonuclar.append([[anlik_yol_list[i]],["Yanlis"],[(gps_latitude,gps_longitude)],[(lat_tahmin,long_tahmin)]])
                cv2.circle(img_temp, centerOfCircle, radius, (0,0,255), 25) 
                
            fk.dosyaya_yaz(sonuclar,dogru_tahmin,yanlis_tahmin) 
                
           
            
            
            
            #img_temp = cv2.cvtColor(img_temp, cv2.COLOR_GRAY2BGR)
                    
            #cv2.rectangle(img_temp, top_left1, bottom_right1,(0,0,255),25)
            #cv2.rectangle(img_temp, top_left2, bottom_right2,(0,255,0),25)
            #cv2.rectangle(img_temp, top_left3, bottom_right3,(255,0,0),25)
            
            #cv2.circle(img_temp, centerOfCircle, radius, (0,255,255), 25)   #tahmini konumu veren nokta
            #cv2.circle(img_temp,(knm[1],knm[0]),radius,(0,255,255), 25)                   #gerçek konumu gösteren nokta
                    #plt.figure()
            
            
            #çizgi çizdirmekiçin yazulır
            """
            if i==0:
                oncesi=(knm[1],knm[0])
            else:
                cv2.line(img_temp, oncesi, (knm[1],knm[0]), (255,0,0), 10) 
                oncesi=(knm[1],knm[0])
                    
            """
                    
                    # plt.imshow(img)
                    # plt.title("Tespit edilen Sonuç"), plt.axis("on")
                    # plt.suptitle(meth)
                    # plt.pause(0.0001)
            #res = cv2.resize(img_temp, dsize=(1000,1000), interpolation=cv2.INTER_CUBIC)
                #cv2.namedWindow("Resized", cv2.WINDOW_NORMAL)
                
            cv2.namedWindow('harita', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('harita', 1000, 1000)    
            cv2.imshow("harita", img_temp)
            
            _ = cv2.waitKey(1)   #☺ekrana verilen haritayı anlık görebilmek için yazılır
            
            
                # cv2.rectangle(img, top_left, bottom_right,(255,0,0),35)
                # plt.figure()
                # plt.subplot(121), plt.imshow(res, cmap = "gray")
                # plt.title("Eşleşen Sonuç"), plt.axis("on")
                # plt.subplot(122), plt.imshow(img)
                # plt.title("Tespit edilen Sonuç"), plt.axis("on")
                # plt.suptitle(meth)
                # img = cv2.imread(harita,0)
        
        
        sonuclar_dosya = open("modele_gore_sonuclar.txt", "a+")
       # sonuclar = np.vstack((sonuclar,dogru_tahmin, yanlis_tahmin)).T
       # print(sonuclar)
        sonuclar_=" "+str(model_list[k])+",  dogru_tahmin: ,"+str(dogru_tahmin)+",  yanlis_tahmin: ,"+str(yanlis_tahmin)+"\n"
     
      
      
        sonuclar_dosya.write(sonuclar_)
        sonuclar_dosya.close()
      
        
        
            
        print("dogru tahmin = ",dogru_tahmin)
        print("yanlış tahmin = ",yanlis_tahmin)
        yuzde=dogru_tahmin/(dogru_tahmin+yanlis_tahmin)
        yuzde=yuzde*100
        print("doğrukuk yüzdesi: {:.2f}".format(yuzde))
