import sys
import os
from pyproj import Transformer
from PyQt5.QtCore import Qt, QRectF, QPointF, QLineF
from PyQt5.QtGui import (QPixmap, QImage, QPainter, QMouseEvent, QColor, 
                         QPolygonF, QPen, QBrush, QFont, QPainterPath)
from enum import Enum, auto
import math
from PyQt5.QtWidgets import (QApplication, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, 
                            QVBoxLayout, QWidget, QPushButton, QFileDialog, QHBoxLayout,
                            QLineEdit, QListWidget, QListWidgetItem, QLabel, QStatusBar)

from PIL import Image
import numpy as np

class CoordinateConverter:
    def __init__(self, image_corners, image_size):
        """
        Args:
            image_corners: 地图四个角落的经纬度坐标 [lon1, lat1, lon2, lat2, lon3, lat3, lon4, lat4]
            image_size: 图像尺寸 (width, height)
        """
        self.image_width, self.image_height = image_size
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        self.inverse_transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

        # 图像像素四角点 (按照与经纬度点顺序一致)
        self.pixel_points = np.array([
            [0, image_size[1]],               # 左上
            [image_size[0], image_size[1]],  # 右上
            [image_size[0], 0],              # 右下
            [0, 0],                          # 左下
        ], dtype=np.float64)

        # 经纬度转为墨卡托坐标
        self.geo_points = []
        for i in range(0, len(image_corners), 2):
            lon, lat = image_corners[i], image_corners[i + 1]
            x, y = self.transformer.transform(lon, lat)
            self.geo_points.append([x, y])
        self.geo_points = np.array(self.geo_points, dtype=np.float64)

        # 计算仿射变换矩阵：从像素坐标 → 墨卡托坐标
        self.affine_matrix = self.compute_affine(self.pixel_points, self.geo_points)

    def compute_affine(self, src_pts, dst_pts):
        """基于4个点对计算仿射变换矩阵"""
        A = []
        B = []
        for i in range(4):
            x, y = src_pts[i]
            u, v = dst_pts[i]
            A.extend([[x, y, 1, 0, 0, 0],
                      [0, 0, 0, x, y, 1]])
            B.extend([u, v])
        A = np.array(A)
        B = np.array(B)
        affine_params, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        return affine_params.reshape(2, 3)

    def pixel_to_lonlat(self, x, y):
        """像素坐标 → 经纬度"""
        pixel = np.array([x, y, 1])
        mercator = self.affine_matrix @ pixel
        lon, lat = self.inverse_transformer.transform(mercator[0], mercator[1])
        return lon, lat

Image.MAX_IMAGE_PIXELS = None

# 物体标签列表
object_labels = [
    'Boeing737', 'Boeing777', 'Boeing747', 'Boeing787', 'A321', 'A220', 'A330', 'A350', 
    'C919', 'ARJ21', 'other airplane', 'Passenger Ship', 'Motorboat', 'Fishing Boat', 
    'Tugboat', 'Engineering Ship', 'Liquid Cargo Ship', 'Dry Cargo Ship', 'Warship', 'other ship', 
    'Small Car', 'Bus', 'Cargo Truck', 'Dump Truck', 'Van', 'Trailer', 'Tractor', 'Truck Tractor', 
    'Excavator', 'other vehicle', 'Baseball Field', 'Basketball Court', 'Football Field', 
    'Tennis Court', 'Roundabout', 'Intersection', 'Bridge'
]

class MeasureMode(Enum):
    NONE = auto()
    DISTANCE = auto()
    ANGLE = auto()

class ImageViewer(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.coord_converter = None
        self.status_bar = None
        self.measure_mode = MeasureMode.NONE
        self.measure_points = []
        self.measure_items = []
        self.temp_line = None
        self.temp_arc = None
        self.total_distance = 0.0
        self.drag_start_pos = None
        self.has_moved = False
        
        # 初始化视图设置
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setScene(scene)

        self.zoom_factor = 1.2
        self.max_zoom = 8
        self.min_zoom = 0.2
        self.current_zoom = 1
        
    def set_coord_converter(self, converter):
        self.coord_converter = converter
        
    def set_status_bar(self, status_bar):
        self.status_bar = status_bar
        
    def mouseMoveEvent(self, event):
        if self.drag_start_pos:
            # 检查鼠标是否移动超过阈值(5像素)
            if (event.pos() - self.drag_start_pos).manhattanLength() > 5:
                self.has_moved = True
                # 在拖动状态下清除临时线
                if self.temp_line:
                    self.scene().removeItem(self.temp_line)
                    self.temp_line = None
                
        if self.coord_converter and self.status_bar and not self.has_moved:
            # 获取鼠标在图片上的坐标
            scene_pos = self.mapToScene(event.pos())
            if 0 <= scene_pos.x() < self.scene().width() and 0 <= scene_pos.y() < self.scene().height():
                # 转换为经纬度
                lon, lat = self.coord_converter.pixel_to_lonlat(scene_pos.x(), scene_pos.y())
                self.status_bar.showMessage(f"经度: {lon:.6f}°, 纬度: {lat:.6f}°")
                
                # 在测量模式下显示临时线段和距离（仅当没有拖动时）
                if self.measure_mode != MeasureMode.NONE and self.measure_points:
                    scene = self.scene()
                    
                    if self.measure_mode == MeasureMode.DISTANCE:
                        if self.temp_line:
                            scene.removeItem(self.temp_line)
                        
                        # 绘制从最后一个测量点到当前鼠标位置的临时线
                        last_point = self.measure_points[-1]
                        line = QLineF(last_point, scene_pos)
                        self.temp_line = scene.addLine(line, QPen(QColor(255, 0, 0, 150), 1))
                        
                        # 计算并显示临时距离
                        pixel_dist = line.length()
                        lon1, lat1 = self.coord_converter.pixel_to_lonlat(last_point.x(), last_point.y())
                        lon2, lat2 = self.coord_converter.pixel_to_lonlat(scene_pos.x(), scene_pos.y())
                        
                        # 使用Web墨卡托投影计算真实距离
                        x1, y1 = self.coord_converter.transformer.transform(lon1, lat1)
                        x2, y2 = self.coord_converter.transformer.transform(lon2, lat2)
                        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)  # 米
                        
                        # 显示临时距离
                        if dist >= 1000:
                            dist_text = f"临时: {dist/1000:.3f} km"
                        else:
                            dist_text = f"临时: {dist:.1f} m"
                        
                        self.status_bar.showMessage(
                            f"经度: {lon:.6f}°, 纬度: {lat:.6f}° | {dist_text}"
                        )
                    
                    elif self.measure_mode == MeasureMode.ANGLE and len(self.measure_points) >= 1:
                        if self.temp_line:
                            scene.removeItem(self.temp_line)
                        
                        # 如果有1个点，绘制从该点到鼠标位置的临时线
                        if len(self.measure_points) == 1:
                            line = QLineF(self.measure_points[0], scene_pos)
                            self.temp_line = scene.addLine(line, QPen(QColor(0, 0, 255, 150), 1))
                        # 如果有2个点以上，绘制从最后一个点到鼠标位置的临时线
                        elif len(self.measure_points) >= 2:
                            line = QLineF(self.measure_points[-1], scene_pos)
                            self.temp_line = scene.addLine(line, QPen(QColor(0, 0, 255, 150), 1))
                        
                        # 显示临时角度信息
                        if len(self.measure_points) >= 2:
                            vec1 = QLineF(self.measure_points[-2], self.measure_points[-1])
                            vec2 = QLineF(self.measure_points[-1], scene_pos)
                            angle = vec1.angleTo(vec2)
                            if angle > 180:
                                angle = 360 - angle
                            self.status_bar.showMessage(
                                f"经度: {lon:.6f}°, 纬度: {lat:.6f}° | 临时角度: {angle:.1f}°"
                            )
                    
        super().mouseMoveEvent(event)

    def wheelEvent(self, event):
        factor = 1.0
        if event.angleDelta().y() > 0:
            factor = self.zoom_factor
        else:
            factor = 1 / self.zoom_factor

        self.scale(factor, factor)
        self.current_zoom *= factor
        if self.current_zoom > self.max_zoom:
            self.resetTransform()
            self.scale(self.max_zoom, self.max_zoom)
        elif self.current_zoom < self.min_zoom:
            self.resetTransform()
            self.scale(self.min_zoom, self.min_zoom)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drag_start_pos = event.pos()
            self.has_moved = False
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            # 如果不是拖动，执行标记点
            if not self.has_moved and self.measure_mode != MeasureMode.NONE:
                scene_pos = self.mapToScene(event.pos())
                if 0 <= scene_pos.x() < self.scene().width() and 0 <= scene_pos.y() < self.scene().height():
                    if self.measure_mode == MeasureMode.DISTANCE:
                        self.handle_distance_measure(scene_pos)
                    elif self.measure_mode == MeasureMode.ANGLE:
                        self.handle_angle_measure(scene_pos)

            self.setDragMode(QGraphicsView.NoDrag)  # 恢复默认状态
            self.drag_start_pos = None
            self.has_moved = False  # ✅ 加上这行
        super().mouseReleaseEvent(event)

    def handle_distance_measure(self, scene_pos):
        scene = self.scene()
        
        # 添加测量点
        self.measure_points.append(scene_pos)
        
        # 绘制红点
        pen = QPen(QColor(255, 0, 0))
        brush = QBrush(QColor(255, 0, 0))
        point = scene.addEllipse(scene_pos.x()-3, scene_pos.y()-3, 6, 6, pen, brush)
        self.measure_items.append(point)
        
        # 如果有多个点，绘制线段
        if len(self.measure_points) > 1:
            # 移除临时线
            if self.temp_line:
                scene.removeItem(self.temp_line)
                self.temp_line = None
            
            # 绘制固定线
            line = QLineF(self.measure_points[-2], self.measure_points[-1])
            line_item = scene.addLine(line, QPen(QColor(255, 0, 0), 1))
            self.measure_items.append(line_item)
            
            # 计算并显示距离
            pixel_dist = line.length()
            lon1, lat1 = self.coord_converter.pixel_to_lonlat(
                self.measure_points[-2].x(), self.measure_points[-2].y())
            lon2, lat2 = self.coord_converter.pixel_to_lonlat(
                self.measure_points[-1].x(), self.measure_points[-1].y())
            
            # 使用Web墨卡托投影计算真实距离
            x1, y1 = self.coord_converter.transformer.transform(lon1, lat1)
            x2, y2 = self.coord_converter.transformer.transform(lon2, lat2)
            dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)  # 米
            
            self.total_distance += dist
            
            # 显示距离标签
            if dist >= 1000:
                dist_text = f"{dist/1000:.3f} km"
            else:
                dist_text = f"{dist:.1f} m"
            
            if len(self.measure_points) > 1:
                total_text = f" (总: {self.total_distance/1000:.3f} km)" if self.total_distance >= 1000 else f" (总: {self.total_distance:.1f} m)"
                dist_text += total_text
            
            font = QFont("Arial", 10)
            text_item = scene.addText(dist_text, font)
            text_item.setPos(scene_pos.x()+10, scene_pos.y()+10)
            text_item.setDefaultTextColor(QColor(255, 0, 0))
            self.measure_items.append(text_item)

    def handle_angle_measure(self, scene_pos):
        scene = self.scene()
        self.measure_points.append(scene_pos)
        # 画点
        pen = QPen(QColor(0,0,255)); brush = QBrush(QColor(0,0,255))
        pt = scene.addEllipse(scene_pos.x()-3, scene_pos.y()-3, 6, 6, pen, brush)
        self.measure_items.append(pt)

        # 如果累积到第三个点，画最终的角
        if len(self.measure_points) >= 3:
            # 清掉临时线
            if self.temp_line:
                scene.removeItem(self.temp_line)
                self.temp_line = None

            # A=p1, B=p2(顶点), C=p3
            p1, p2, p3 = self.measure_points[-3], self.measure_points[-2], self.measure_points[-1]

            # 两条边
            line1 = QLineF(p2, p1)
            line2 = QLineF(p2, p3)
            self.measure_items.append(scene.addLine(line1, QPen(QColor(0,0,255),1)))
            self.measure_items.append(scene.addLine(line2, QPen(QColor(0,0,255),1)))

            # 无向夹角
            angle = line1.angleTo(line2)
            if angle > 180:
                angle = 360 - angle

            # 判断方向：叉积 >0 则 CCW，否则 CW
            cross = (p1.x()-p2.x())*(p3.y()-p2.y()) - (p1.y()-p2.y())*(p3.x()-p2.x())
            sweep = -angle if cross >=0 else angle  # 增加负号

            # 起始角度：line1.angle() 给出“line1 相对 x 轴 CCW 的角度”
            start_angle = line1.angle()

            # 画圆弧
            radius = min(line1.length(), line2.length()) / 2
            rect = QRectF(p2.x()-radius, p2.y()-radius, radius*2, radius*2)
            path = QPainterPath()
            path.arcMoveTo(rect, start_angle)
            path.arcTo(rect, start_angle, sweep)
            arc = scene.addPath(path, QPen(QColor(0,0,255)), QBrush(Qt.transparent))
            self.measure_items.append(arc)

            # 角度文字
            text = scene.addText(f"{angle:.1f}°", QFont("Arial",10))
            text.setPos(p2.x()+10, p2.y()+10)
            text.setDefaultTextColor(QColor(0,0,255))
            self.measure_items.append(text)

import json

def load_geojson_corners(geojson_path):
    with open(geojson_path, 'r') as f:
        geojson = json.load(f)
    coords = geojson['features'][0]['geometry']['coordinates'][0]
    # 展平为 [lon1, lat1, lon2, lat2, ...]
    flat_coords = [coord for point in coords for coord in point]
    return flat_coords

class ImageScene(QGraphicsScene):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.pixmap_item = None
        self.detection_boxes = []
        self.labels = []
        self.load_image()

    def load_image(self):
        image = Image.open(self.image_path)
        image = image.convert("RGB")
        qt_image = QImage(image.tobytes(), image.width, image.height, image.width * 3, QImage.Format_RGB888)
        pixmap = QPixmap(qt_image)
        
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        self.addItem(self.pixmap_item)

    def load_detections(self, file_path):
        detections = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                id = int(parts[0])
                conf = parts[1]
                coords = parts[2:]
                detections.append([id, conf] + coords)
        return detections

    def draw_detection_boxes(self, detections):
        self.clear_detection_boxes()
        for detection in detections:
            id, conf, x1, y1, x2, y2, x3, y3, x4, y4 = detection
            polygon = QPolygonF([QPointF(x1, y1), QPointF(x2, y2), QPointF(x3, y3), QPointF(x4, y4)])
            pen = QPen(QColor(255, 0, 0))
            brush = QBrush(Qt.transparent)
            polygon_item = self.addPolygon(polygon, pen, brush)
            self.detection_boxes.append(polygon_item)

    def draw_labels(self, detections):
        self.clear_labels()
        for detection in detections:
            id, conf, x1, y1, x2, y2, x3, y3, x4, y4 = detection
            label = object_labels[id]
            text = f"{label} {conf:.2f}"
            font = QFont("Arial", 10)
            text_item = self.addText(text, font)
            text_item.setPos(x1, y1)
            self.labels.append(text_item)

    def clear_detection_boxes(self):
        for box in self.detection_boxes:
            self.removeItem(box)
        self.detection_boxes.clear()

    def clear_labels(self):
        for label in self.labels:
            self.removeItem(label)
        self.labels.clear()

class ImageViewerApp(QWidget):
    def __init__(self, base_path):
        super().__init__()
        image_path = os.path.join(base_path, os.path.basename(base_path) + ".tif")
        geojson_path = os.path.join(base_path, os.path.basename(base_path) + ".geojson")

        self.image_corners = load_geojson_corners(geojson_path)
        self.setWindowTitle('遥感图片查看器')
        self.setGeometry(100, 100, 1200, 800)

        # 初始化场景和视图
        scene = ImageScene(image_path=image_path)
        self.view = ImageViewer(scene)
        
        # 创建主垂直布局
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # 创建内容区域水平布局
        self.content_layout = QHBoxLayout()
        self.main_layout.addLayout(self.content_layout)

        self.init_ui()

        self.status_bar = QStatusBar()
        self.view.set_status_bar(self.status_bar)

        image = Image.open(image_path)
        self.coord_converter = CoordinateConverter(self.image_corners, (image.width, image.height))
        self.view.set_coord_converter(self.coord_converter)

        self.main_layout.addWidget(self.status_bar)

        self.detections = []
        self.filtered_detections = []

    def init_ui(self):
        # 左侧布局（原功能）
        left_layout = QVBoxLayout()
        self.load_button = QPushButton("载入标签")
        self.show_box_button = QPushButton("显示检测框") 
        self.show_label_button = QPushButton("显示标签")
        self.measure_distance_button = QPushButton("测距离")
        self.measure_angle_button = QPushButton("任意角")
        
        left_layout.addWidget(self.load_button)
        left_layout.addWidget(self.show_box_button)
        left_layout.addWidget(self.show_label_button)
        left_layout.addWidget(self.measure_distance_button)
        left_layout.addWidget(self.measure_angle_button)
        left_layout.addWidget(self.view, stretch=1)
        
        # 右侧布局（新增搜索功能）
        right_layout = QVBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("输入关键词搜索...")
        self.results_list = QListWidget()
        
        right_layout.addWidget(self.search_input)
        right_layout.addWidget(self.results_list)
        
        # 信号连接
        self.load_button.clicked.connect(self.load_labels)
        self.show_box_button.clicked.connect(self.show_detection_boxes)
        self.show_label_button.clicked.connect(self.show_labels)
        self.measure_distance_button.clicked.connect(self.toggle_distance_measure)
        self.measure_angle_button.clicked.connect(self.toggle_angle_measure)
        self.search_input.textChanged.connect(self.update_search_results)
        self.results_list.itemClicked.connect(self.on_result_clicked)
        
        # 布局比例设置
        self.content_layout.addLayout(left_layout, stretch=4)
        self.content_layout.addLayout(right_layout, stretch=1)

    def load_labels(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择标签文件", "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            scene = self.view.scene()
            self.detections = scene.load_detections(file_path)
            self.update_search_results()  # 载入新标签后自动刷新搜索结果

    def show_detection_boxes(self):
        if self.detections:
            self.view.scene().draw_detection_boxes(self.detections)

    def show_labels(self):
        if self.detections:
            self.view.scene().draw_labels(self.detections)

    def update_search_results(self):
        keyword = self.search_input.text().strip().lower()
        self.results_list.clear()
        self.filtered_detections = []
        
        if not keyword or not self.detections:
            return
            
        for detection in self.detections:
            label_id = detection[0]
            label_name = object_labels[label_id].lower()
            if keyword in label_name:
                self.filtered_detections.append(detection)
                item_text = f"{object_labels[label_id]} {detection[1]:.2f}"
                self.results_list.addItem(QListWidgetItem(item_text))

    def on_result_clicked(self, item):
        row = self.results_list.row(item)
        if 0 <= row < len(self.filtered_detections):
            detection = self.filtered_detections[row]
            coords = detection[2:10]
            x_center = sum(coords[0::2]) / 4  # 计算x坐标平均值
            y_center = sum(coords[1::2]) / 4  # 计算y坐标平均值
            self.view.centerOn(x_center, y_center)
            self.view.setFocus()

    def toggle_distance_measure(self):
        if self.view.measure_mode == MeasureMode.DISTANCE:
            self.view.measure_mode = MeasureMode.NONE
            self.clear_measurements()
        else:
            self.view.measure_mode = MeasureMode.DISTANCE
            self.clear_measurements()
            self.measure_distance_button.setText("退出测距")
            self.measure_angle_button.setEnabled(False)

    def toggle_angle_measure(self):
        if self.view.measure_mode == MeasureMode.ANGLE:
            self.view.measure_mode = MeasureMode.NONE
            self.clear_measurements()
        else:
            self.view.measure_mode = MeasureMode.ANGLE
            self.clear_measurements()
            self.measure_angle_button.setText("退出测角")
            self.measure_distance_button.setEnabled(False)

    def clear_measurements(self):
        scene = self.view.scene()
        for item in self.view.measure_items:
            scene.removeItem(item)
        if self.view.temp_line:
            scene.removeItem(self.view.temp_line)
            self.view.temp_line = None
        if self.view.temp_arc:
            scene.removeItem(self.view.temp_arc)
            self.view.temp_arc = None
        self.view.measure_items.clear()
        self.view.measure_points.clear()
        self.view.total_distance = 0.0
        self.measure_distance_button.setText("测距离")
        self.measure_angle_button.setText("任意角")
        self.measure_distance_button.setEnabled(True)
        self.measure_angle_button.setEnabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    base_path = './datasets/baiyun_airport'
    viewer = ImageViewerApp(base_path)
    viewer.show()
    sys.exit(app.exec_())
