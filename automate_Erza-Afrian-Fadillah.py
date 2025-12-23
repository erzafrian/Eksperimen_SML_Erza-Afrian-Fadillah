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

    def load_data(self):
        """Memuat data dari file CSV."""
        try:
            self.df = pd.read_csv(self.file_path)
            print("Data berhasil dimuat.")
        except FileNotFoundError:
            raise Exception("File tidak ditemukan. Pastikan path file benar.")

    def clean_data(self):
        """Menghapus duplikat dan nilai kosong."""
        initial_shape = self.df.shape
        self.df.drop_duplicates(inplace=True)
        self.df.dropna(inplace=True)
        print(f"Data cleaning selesai. Rows dropped: {initial_shape[0] - self.df.shape[0]}")

    def drop_unnecessary_columns(self):
        """Menghapus kolom yang tidak diperlukan sesuai eksperimen."""
        columns_to_drop = [
            'Order_ID', 'Customer_ID', 'Date', 'Age', 'Gender', 'City',
            'Product_Category', 'Unit_Price', 'Quantity', 'Total_Amount',
            'Payment_Method', 'Device_Type', 'Is_Returning_Customer'
        ]
        self.df = self.df.drop(columns=columns_to_drop, axis=1, errors='ignore')
        print(f"Kolom tidak relevan dihapus. Sisa kolom: {list(self.df.columns)}")

    def remove_outliers_iqr(self):
        """
        Menghapus outlier menggunakan metode IQR pada kolom numerik 
        (kecuali target).
        """
        # Identifikasi kolom numerik selain target (Customer_Rating)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'Customer_Rating']
        
        initial_rows = self.df.shape[0]
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Filter data
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
            
        print(f"Outlier removal selesai. Rows removed: {initial_rows - self.df.shape[0]}")

    def split_data(self, target_column='Customer_Rating', test_size=0.2, random_state=42):
        """Membagi data menjadi set training dan testing."""
        X = self.df.drop(target_column, axis=1)
        y = self.df[target_column]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"Data split selesai. Train size: {self.X_train.shape}, Test size: {self.X_test.shape}")

    def scale_features(self):
        """Melakukan standarisasi fitur (StandardScaler)."""
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print("Fitur telah distandarisasi.")

    def run_pipeline(self):
        """Menjalankan seluruh tahapan preprocessing secara berurutan."""
        print("Memulai Otomasi Preprocessing")
        self.load_data()
        self.clean_data()
        self.drop_unnecessary_columns()
        self.remove_outliers_iqr() 
        self.split_data()
        self.scale_features()
        print("Preprocessing Selesai\n")
        
        return self.X_train, self.X_test, self.y_train, self.y_test

# Fungsi wrapper agar mudah dipanggil dari script lain
def get_preprocessed_data(file_path):
    preprocessor = DataPreprocessor(file_path)
    return preprocessor.run_pipeline()

if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test = get_preprocessed_data('Eksperimen_SML_Erza-Afrian-Fadillah\customer_behavior.csv')
        print("Contoh data training (scaled):")
        print(X_train[:5])
        print("Contoh target training:")
        print(y_train[:5])
    except Exception as e:
        print(f"Error: {e}")