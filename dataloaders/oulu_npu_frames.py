import os, sys
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from . import utils_dataloaders as ud


class OULU_NPU_FRAMES(Dataset):
    def __init__(self, root_dir, protocol_id, frames_path, img_size, part='train', local_rank=0, transform=None):
        super(OULU_NPU_FRAMES, self).__init__()
        self.img_size = img_size
        self.protocols_path = os.path.join(root_dir, 'Protocols', 'Protocol_'+str(protocol_id))
        
        if part == 'train':
            self.root_dir_part = os.path.join(root_dir, 'train')
            self.protocol_file_path = os.path.join(self.protocols_path, 'Train.txt')
        elif part == 'val' or part == 'validation' or part == 'dev' or part == 'development':
            self.root_dir_part = os.path.join(root_dir, 'dev')
            self.protocol_file_path = os.path.join(self.protocols_path, 'Dev.txt')
        elif part == 'test':
            self.root_dir_part = os.path.join(root_dir, 'test')
            self.protocol_file_path = os.path.join(self.protocols_path, 'Test.txt')
        else:
            raise Exception(f'Error, dataset partition not recognized: \'{part}\'')

        self.frames_path_part = self.root_dir_part.replace(root_dir, root_dir)

        self.protocol_data = ud.load_file_protocol(self.protocol_file_path)

        self.rgb_file_ext = '.png'
        self.samples_list = ud.make_samples_list(self.protocol_data, self.frames_path_part, self.rgb_file_ext)
        
        assert len(self.protocol_data) == len(self.samples_list), 'Error, len(self.protocol_data) must be equals to len(self.samples_list)'


    def normalize_img(self, img):
        img = np.transpose(img, (2, 0, 1))  # from (224,224,3) to (3,224,224)
        img = ((img/255.)-0.5)/0.5
        # print('img:', img)
        # sys.exit(0)
        return img

    
    def load_img(self, img_path):
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # img_rgb = np.asarray(Image.open(img_path))
        # print('img:', img)
        # sys.exit(0)
        if (img_rgb.shape[0], img_rgb.shape[1]) != (self.img_size, self.img_size):
            img_rgb = cv2.resize(img_rgb, dsize=(self.img_size,self.img_size), interpolation=cv2.INTER_AREA)
        return img_rgb.astype(np.float32)


    def normalize_pc(self, pc):
        pc = (pc - pc.min()) / (pc.max() - pc.min())
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc


    def __getitem__(self, index):
        # idx = self.imgidx[index]
        # s = self.imgrec.read_idx(idx)
        # header, img = mx.recordio.unpack(s)
        # label = header.label
        # if not isinstance(label, numbers.Number):
        #     label = label[0]
        # label = torch.tensor(label, dtype=torch.long)
        # sample = mx.image.imdecode(img).asnumpy()
        # if self.transform is not None:
        #     sample = self.transform(sample)
        # return sample, label

        # Bernardo
        # img_path, pc_path, label = self.samples_list[index]
        img_path, label = self.samples_list[index]

        if img_path.endswith('.jpg') or img_path.endswith('.jpeg') or img_path.endswith('.png'):
            rgb_data = self.load_img(img_path)
            rgb_data = self.normalize_img(rgb_data)

        # if pc_path.endswith('.obj'):
        #     pc_data = self.read_obj(pc_path)['vertices']
        # elif pc_path.endswith('.npy'):
        #     pc_data = np.load(pc_path)
        #     # pc_data = np.load(pc_path).astype(np.float32)

        # pc_data = self.normalize_pc(pc_data)
        # pc_data = self.sample_points(pc_data, n=2500)
        #
        # if label == 0:
        #     pc_data = self.flat_pc_axis_z(pc_data)
        
        # save_path = f'./pointcloud_index={index}_label={label}.obj'
        # self.write_obj(save_path, pc_data)
        
        return (rgb_data, label)
        # return (rgb_data, pc_data, label)
        # return (img_path, rgb_data, pc_path, pc_data, label)


    def __len__(self):
        # return len(self.imgidx)       # original
        return len(self.samples_list)   # Bernardo