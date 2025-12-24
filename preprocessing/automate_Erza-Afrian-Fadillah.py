import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

    def load_data(self):
        """Memuat data dari file CSV."""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Data berhasil dimuat dari {self.file_path}")
        except FileNotFoundError:
            print(f"Error: File '{self.file_path}' tidak ditemukan!")
            raise

    def clean_data(self):
        """Menghapus duplikat dan nilai kosong."""
        self.df.drop_duplicates(inplace=True)
        self.df.dropna(inplace=True)
        print("Data cleaning selesai.")

    def drop_unnecessary_columns(self):
        """Menghapus kolom yang tidak diperlukan sesuai eksperimen."""
        columns_to_drop = [
            'Order_ID', 'Customer_ID', 'Date', 'Age', 'Gender', 'City',
            'Product_Category', 'Unit_Price', 'Quantity', 'Total_Amount',
            'Payment_Method', 'Device_Type', 'Is_Returning_Customer'
        ]
        self.df = self.df.drop(columns=columns_to_drop, axis=1, errors='ignore')
        print(f"Kolom dibuang. Sisa kolom: {list(self.df.columns)}")

    def remove_outliers_iqr(self):
        """Menghapus outlier menggunakan metode IQR."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'Customer_Rating']
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            self.df = self.df[(self.df[col] >= Q1 - 1.5 * IQR) & (self.df[col] <= Q3 + 1.5 * IQR)]
        print("Outlier berhasil ditangani.")

    def split_data(self, target_column='Customer_Rating'):
        """Membagi data menjadi set training dan testing."""
        X = self.df.drop(target_column, axis=1)
        y = self.df[target_column]
        self.feature_names = X.columns
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print("Data split selesai.")

    def scale_features(self):
        """Melakukan standarisasi fitur."""
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print("Fitur telah distandarisasi.")

    def save_processed_data(self, output_filename='customer_behavior_preprocessing.csv'):
        """Menggabungkan data training yang sudah di-scale dan menyimpannya ke CSV."""
        # Membuat DataFrame dari X_train 
        processed_df = pd.DataFrame(self.X_train, columns=self.feature_names)
        
        # Menambahkan kembali kolom target 
        processed_df['Customer_Rating'] = self.y_train.values
        
        # MENYIMPAN KE FILE
        processed_df.to_csv(output_filename, index=False)
        print(f"Data hasil preprocessing disimpan ke: {output_filename}")
        

    def run_pipeline(self):
        """Menjalankan seluruh tahapan preprocessing secara otomatis."""
        print("\nMemulai Otomasi Preprocessing")
        self.load_data()
        self.clean_data()
        self.drop_unnecessary_columns()
        self.remove_outliers_iqr()
        self.split_data()
        self.scale_features()
        self.save_processed_data() # Memanggil fungsi simpan
        print("Proses Selesai & Data Siap Digunakan\n")
        
        return self.X_train, self.X_test, self.y_train, self.y_test

# Fungsi Utama
if __name__ == "__main__":
    input_file = 'Eksperimen_SML_Erza-Afrian-Fadillah/customer_behavior_cleaned.csv' 
    
    # Jalankan otomatisasi
    try:
        preprocessor = DataPreprocessor(input_file)
        X_train, X_test, y_train, y_test = preprocessor.run_pipeline()
    except Exception as e:
        print(f"Error: {e}")