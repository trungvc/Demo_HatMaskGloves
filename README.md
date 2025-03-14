# Giới thiệu

Đây là dự án sử dụng mô hình YOLO kết hợp với BoT-SORT để phát hiện và theo dõi các đối tượng trong ảnh/video, cụ thể là ba lớp: mũ (hat), khẩu trang (mask), và găng tay (gloves).


## Cấu Trúc Dự Án
```
📂 Demo_HatMaskGloves
├── 📂 model
│   ├── best_model.pt  #Model segmentation khẩu trang, găng tay, mũ
│   ├── yolo11n.pt      #Model detect người
├── 📂 src
│   ├── Thread_Segment.py    #Luồng xử lý segment khẩu trang, găng tay, mũ
│   ├── Thread_capture.py    #Luồng đọc video đầu vào và lưu frame vào Queue
│   ├── Thread_detect.py    #Luồng lấy frame từ trong Queue ra để xử lý detect và tracking
│   ├── Thread_stream.py    #Luồng stream frame đã xử lý
│   ├── config.py          #Thay đổi link model hay video demo
│   ├── main.py            #Chạy chương trình
│   ├── main_controller.py      #Điều khiển luồng
├── .gitignore
├── requirements.txt    #File cài thư viện
```
