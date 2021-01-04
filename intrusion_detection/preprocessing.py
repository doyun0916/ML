import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

def scaling(d):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(d)
    return data_scaled

def pca_intrusion(d, opt):
    pc = PCA(n_components=20)  # 주성분을 몇개로 할지 결정
    components = pc.fit_transform(d)
    if opt==0:
        plt.plot(np.cumsum(pc.explained_variance_ratio_))           # finding number of components
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance in intrusion data')
        plt.show()
        print("Intrusion PCA Explanation rate: ", sum(pc.explained_variance_ratio_), "\n")  # 최고 수치 나타내는것 find
    components_data = pd.DataFrame(data=components, columns=['pc1', 'pc2', 'pc3', 'pc4', 'pc5',
                                                             'pc6', 'pc7', 'pc8', 'pc9', 'pc10',
                                                             'pc11', 'pc12', 'pc13', 'pc14', 'pc15',
                                                             'pc16', 'pc17', 'pc18', 'pc19', 'pc20'])
    return components_data

def pca_traffic(d, opt):
    pc = PCA(n_components=15)  # 주성분을 몇개로 할지 결정
    components = pc.fit_transform(d)
    if opt==0:
        plt.plot(np.cumsum(pc.explained_variance_ratio_))          # finding number of components
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance in traffic data')
        plt.show()
        print("Traffic PCA Explanation rate: ", sum(pc.explained_variance_ratio_), "\n")  # 최고 수치 나타내는것 find
    components_data = pd.DataFrame(data=components, columns=['pc1', 'pc2', 'pc3', 'pc4', 'pc5',
                                                             'pc6', 'pc7', 'pc8', 'pc9', 'pc10',
                                                             'pc11', 'pc12', 'pc13', 'pc14', 'pc15'])
    return components_data

def preprocess(data, ds, opt=0):
    y = data['class']
    x = data.drop(columns=['class'])
    scaled_data = scaling(x)
    if ds == 0:
        preprocessed_data_x = pca_intrusion(scaled_data, opt)
    else:
        preprocessed_data_x = pca_traffic(scaled_data, opt)
    return preprocessed_data_x, y

def timestamp_conv(x):
    x = x.replace("-", "").replace(":", "").replace(" ", "")
    return float(x)


def ip_conv(x):
    x = x.replace(".", "").replace(" ", "")
    return int(x)


def trafdata_filter(data):
    data["DoH"].replace({True: 0, False: 1}, inplace=True)
    t = data['TimeStamp'].apply(timestamp_conv)
    d = data['DestinationIP'].apply(ip_conv)
    i = data['SourceIP'].apply(ip_conv)
    data.drop(columns=['TimeStamp', 'DestinationIP', 'SourceIP'], inplace=True)

    data = pd.concat([data, t, d, i], axis=1)

    data.dropna(inplace=True)
    data.dropna(inplace=True, axis=1)
    return data


##################################### Data preprocessiong ##################################################

### intrusion data
def intrusion_data():
    intrusion = pd.read_csv("./intrusion_detection/Train_data.csv")        # read intrusion data
    intrusion_len = len(intrusion)
    print("Intrusion data 수: ", intrusion_len)
    print("Intrusion data types:\n", intrusion.dtypes, "\n")

    intrusion["class"].replace({"normal": 0, "anomaly": 1}, inplace=True)   # 가변수 처리
    intrusion['protocol_type'] = intrusion['protocol_type'].astype('category').cat.codes.astype(np.int32) * 50
    intrusion['service'] = intrusion['service'].astype('category').cat.codes.astype(np.int32) * 50
    intrusion['flag'] = intrusion['flag'].astype('category').cat.codes.astype(np.int32) * 50

    # 결측치 제거
    print("Intrusion data 결측치:\n", intrusion.isnull().sum(), "\n")
    intrusion.dropna(inplace=True)
    intrusion.dropna(inplace=True, axis=1)
    print("Intrusion data 기존 수 : ", intrusion_len)
    print("Intrusion data 결측치 제거 후 수 : ", len(intrusion), "\n")

    preprocessed_data, y = preprocess(intrusion, 0)
    intru_x_train, intru_x_test, intru_y_train, intru_y_test = train_test_split(preprocessed_data, y, test_size=0.3, train_size=0.7, random_state=42)
    print("Intrusion x_train data 수: ", len(intru_x_train))
    print("Intrusion x_test data 수: ", len(intru_x_test))
    print("Intrusion y_train data 수: ", len(intru_y_train))
    print("Intrusion y_test data 수: ", len(intru_y_test))

    return intrusion, intru_x_train, intru_x_test, intru_y_train, intru_y_test


### traffic data
def traffic_data():
    benign = pd.read_csv("./DNS malign/benign-csvs/benign-chrome.csv")         # read benign network traffic data
    benign_len = len(benign)
    print("\n\nBenign traffic data 수: ", benign_len)
    print("Benign traffic data types:\n", benign.dtypes, "\n")

    benign['class'] = 0                                                        # 가변수 처리
    benign["DoH"].replace({True: 0, False: 1}, inplace=True)
    time = benign['TimeStamp'].apply(timestamp_conv)
    des = benign['DestinationIP'].apply(ip_conv)
    sor = benign['SourceIP'].apply(ip_conv)
    benign.drop(columns=['TimeStamp', 'DestinationIP', 'SourceIP'], inplace=True)
    benign = pd.concat([benign, time, des, sor], axis=1)

    # 결측치 제거
    print("Benign data 결측치:\n\n", benign.isnull().sum(), "\n")
    benign.dropna(inplace=True)
    benign.dropna(inplace=True, axis=1)
    print("Benign data 기존 수 : ", benign_len)
    print("Benign data 결측치 제거 후 수 : ", len(benign), "\n")
    b_sam = benign.sample(n=6298)

    mal_dns2tcp = pd.read_csv("./DNS malign/malic-csvs/mal-dns2tcp.csv")        # read malign_tcp network traffic data
    mal_dns2tcp['class'] = 1                                                    # 가변수, 결측치 처리
    mal_dns2tcp_filt = trafdata_filter(mal_dns2tcp)
    t_sam = mal_dns2tcp_filt.sample(n=6298)

    mal_dnscat2 = pd.read_csv("./DNS malign/malic-csvs/mal-dnscat2.csv")        # read malign_cat network traffic data
    mal_dnscat2['class'] = 2                                                    # 가변수, 결측치 처리
    mal_dnscat2_filt = trafdata_filter(mal_dnscat2)
    c_sam = mal_dnscat2_filt.sample(n=6298)

    mal_iodine = pd.read_csv("./DNS malign/malic-csvs/mal-iodine.csv")          # read malign_iodine network traffic data
    mal_iodine['class'] = 3                                                     # 가변수, 결측치 처리
    mal_iodine_filt = trafdata_filter(mal_iodine)
    i_sam = mal_iodine_filt.sample(n=6298)

    traffic_sam = pd.concat([b_sam, t_sam, c_sam, i_sam], ignore_index=True)
    preprocessed_data, y = preprocess(traffic_sam, 1)
    traffic_x_train, traffic_x_test, traffic_y_train, traffic_y_test = train_test_split(preprocessed_data, y, test_size=0.3, train_size=0.7, random_state=42)
    print("traffic x_train data 수: ", len(traffic_x_train))
    print("traffic x_test data 수: ", len(traffic_x_test))
    print("traffic y_train data 수: ", len(traffic_y_train))
    print("traffic y_test data 수: ", len(traffic_y_test))

    return traffic_sam, traffic_x_train, traffic_x_test, traffic_y_train, traffic_y_test


# 실제 적용시, 두개 classification 모델 동시에 사용할 때, test data 만들기
def overall_test_data(intrusion, traffic):
    intrusion_sample = intrusion.sample(n=10000)
    traffic_sample = traffic.sample(n=10000)
    intru_sam_pre_x, intru_sam_pre_y = preprocess(intrusion_sample, 0, 1)
    traf_sam_pre_x, traf_sam_pre_y = preprocess(traffic_sample, 1, 1)
    total_sample_x = pd.concat([intru_sam_pre_x, traf_sam_pre_x], ignore_index=True, axis=1)
    total_sample_y = pd.concat([intru_sam_pre_y, traf_sam_pre_y], ignore_index=True, axis=0)
    total_sample_x = total_sample_x.fillna(0)
    return total_sample_x, total_sample_y
