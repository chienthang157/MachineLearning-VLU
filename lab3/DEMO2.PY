# nhập thư viện
import numpy as np
import pandas as pd
# tạo hàm biến đổi cột định tính, dùng phương pháp one hot
def transform(data, columns_trans): # data dạng dataframe, data_trans là cột cần biến đổi --> dạng Series, nhiều cột cần biến đổi thì bỏ vào list
    for i in columns_trans:
        unique = data[i].unique() + '-' + i # trả lại mảng
        # tạo ma trận 0
        matrix_0 = np.zeros((len(data), len(unique)), dtype = int)
        frame_0 = pd.DataFrame(matrix_0, columns = unique)
        for index, value in enumerate(data[i]):
            frame_0.at[index, value + '-' + i] = 1
        data[unique] = frame_0
    return data # trả lại data truyền vào nhưng đã bị biến đổi
# tạo hàm scale dữ liệu về [0,1] (min max scaler)
def scale_data(data, columns_scale): # columns_scale là cột cần scale, nếu nhiều bỏ vào list ['a', 'b']
    for i in columns_scale:  
        _max = data[i].max()
        _min = data[i].min()
        '''Code here'''
        min_max_scaler = lambda x: round((x - _min) / (_max - _min), 3)
        data[i] = data[i].apply(min_max_scaller)
    return data # --> trả về frame