   # def load_image_custom(self, i, rect_mode=True):
    #     """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
    #     # print("load image custom!!!!!")
    #     im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
    #     rd, f_rd, fn_rd = self.rds[i], self.radar_files[i], self.npy_files_rd[i]
    #     rv, f_rv, fn_rv = self.rvs[i], self.rvm_files[i], self.npy_files_rv[i]
    #     if im is None:  # not cached in RAM
    #         if fn.exists():  # load npy
    #             try:
    #                 im = np.load(fn)
    #             except Exception as e:
    #                 LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
    #                 Path(fn).unlink(missing_ok=True)
    #                 im = cv2.imread(f)  # BGR
    #         else:  # read image
    #             im = cv2.imread(f)  # BGR
    #         if im is None:
    #             raise FileNotFoundError(f"Image Not Found {f}")

    #         h0, w0 = im.shape[:2]  # orig hw
    #         if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
    #             r = self.imgsz / max(h0, w0)  # ratio
    #             if r != 1:  # if sizes are not equal
    #                 w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
    #                 im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
    #         elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
    #             im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

    #         # Add to buffer if training with augmentations
    #         if self.augment:
    #             self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
    #             self.buffer.append(i)
    #             if len(self.buffer) >= self.max_buffer_length:
    #                 j = self.buffer.pop(0)
    #                 self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

    #         return im, (h0, w0), im.shape[:2]

    #     return self.ims[i], self.im_hw0[i], self.im_hw[i]

                     from  n    params  module                                       arguments                     
  0                  -1  1       928  ultralytics.nn.modules.conv.Conv             [3, 32, 3, 2]                 
  1                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
  2                  -1  1     29056  ultralytics.nn.modules.block.C2f             [64, 64, 1, True]             
  3                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  4                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           
  5                  -1  1     36096  ultralytics.nn.modules.block.SCDown          [128, 256, 3, 2]              
  6                  -1  2    788480  ultralytics.nn.modules.block.C2f             [256, 256, 2, True]           
  7                  -1  1    137728  ultralytics.nn.modules.block.SCDown          [256, 512, 3, 2]              
  8                  -1  1    958464  ultralytics.nn.modules.block.C2fCIB          [512, 512, 1, True, True]     
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.PSA             [512, 512]                    
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1    591360  ultralytics.nn.modules.block.C2f             [768, 256, 1]                 
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
 17                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
 20                  -1  1     68864  ultralytics.nn.modules.block.SCDown          [256, 256, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1089536  ultralytics.nn.modules.block.C2fCIB          [768, 512, 1, True, True]     
 23        [16, 19, 22]  1   1645766  ultralytics.nn.modules.head.v10Detect        [9, [128, 256, 512]]    