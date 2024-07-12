import os

def delete_non_matching_files(folder1, folder2):
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))
    
    # folder1'de olup folder2'de olmayan dosyaları sil
    for file in files1 - files2:
        file_path = os.path.join(folder1, file)
        os.remove(file_path)
        print(f"{file_path} silindi.")
    
    # folder2'de olup folder1'de olmayan dosyaları sil
    for file in files2 - files1:
        file_path = os.path.join(folder2, file)
        os.remove(file_path)
        print(f"{file_path} silindi.")

# Kullanım
folder1 = 'C:/Users/SUEN/Desktop/verim/X'  # İlk klasörün yolunu buraya yazın
folder2 = 'C:/Users/SUEN/Desktop/verim/y'  # İkinci klasörün yolunu buraya yazın
delete_non_matching_files(folder1, folder2)
