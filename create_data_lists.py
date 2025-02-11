from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_data='C:\\code\\a-PyTorch-Tutorial-to-Object-Detection-master\\finis_data\\train',
                      test_data='C:\\code\\a-PyTorch-Tutorial-to-Object-Detection-master\\finis_data\\test',
                      output_folder='./')
# write your data path like
"""if __name__ == '__main__':
    create_data_lists(train_data='C:\\code\\...',
                      test_data='C:\\code\\...',
                      output_folder='./')"""
