import flet as ft
from views.history import History_Screen
from views.file_to_doc import File_to_doc_Screen
from views.receive_img import run_web_upload_image
import easyocr
from lib.vietocr.tool.predictor import Predictor
from lib.vietocr.tool.config import Cfg
import re
import repath
import os
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def main(page: ft.Page):
    page.title = "Phần mềm OCR chuyển ảnh thành file Word"

    loading_screen = ft.Container(
                        alignment=ft.alignment.center,
                        content=ft.Column(
                                height = page.window_height,
                                width = page.window_width,
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                controls=[
                                    ft.Container(
                                        content = ft.Row(
                                            height = page.window_height,
                                            width = page.window_width,
                                            alignment = ft.MainAxisAlignment.CENTER,
                                            vertical_alignment=ft.MainAxisAlignment.CENTER,
                                            controls = [
                                                ft.ProgressRing(), 
                                                ft.Text("Vui lòng chờ trong lúc chương trình khởi động...")]))
                                    ],
                            ))
    page.add(loading_screen)

    detect_text_model = easyocr.Reader(['vi'], gpu=False)
    config = Cfg.load_config_from_name(r'vgg_seq2seq') # sử dụng config mặc định của mình 
    config['weights'] = resource_path(r'lib\vietocr\weights\vgg_seq2seq.pth') # đường dẫn đến trọng số đã huấn luyện hoặc comment để sử dụng pretrained model của mình
    config['device'] = 'cpu' # device chạy 'cuda:0', 'cuda:1', 'cpu'
    read_text_detector = Predictor(config)

    page.remove(loading_screen)

    #Pick file from local storage computer
    def pick_files_result(e: ft.FilePickerResultEvent):
        selected_files_path.value = (
            ", ".join(map(lambda f: f.path, e.files)) if e.files else None
        )
        if selected_files_path.value is not None and selected_files_path.value != "":
            selected_files_path.update()
            # print("Selected files:", selected_files_path.value)
            page.go(route="/file_to_doc"+"/"+selected_files_path.value)
            # print(page.__dict__)
            # print(urlparse(ft.TemplateRoute(page.route).route))

    pick_files_dialog = ft.FilePicker(on_result=pick_files_result)
    selected_files_path = ft.Text(value=None)
    page.overlay.append(pick_files_dialog)

    # Dialog for caution open localhost to recieve image through web
    qr_img = ft.Image(
        width=300,
        height=300,
        fit=ft.ImageFit.CONTAIN,
        filter_quality=ft.FilterQuality.HIGH,
    )

    qr_text = ft.Text()

    qr_info = ft.Row(
                alignment=ft.MainAxisAlignment.CENTER,
                controls=[
                    ft.Column(
                        alignment=ft.MainAxisAlignment.CENTER,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        width = 400,
                        controls = [
                            qr_img,
                            qr_text,
                        ]
                    )])

    qr_container = ft.Container(
        margin = 10,
        width = 400,
    )

    def open_qr_modal(e):
        url = run_web_upload_image()
        qr_img.src = resource_path(r"views/latest_qr_code/qr.png")
        qr_text.value = "Hoặc truy cập đường dẫn " + url
        qr_container.content = qr_info
        page.dialog = qr_dlg
        qr_dlg.open = True
        # dlg_modal.open = False
        page.update()

    def close_dlg(e):
        page.dialog = dlg_modal
        dlg_modal.open = False
        qr_dlg.open = False
        page.update()

    def open_dlg_modal_to_recieve_file_from_other_device():
        page.dialog = dlg_modal
        dlg_modal.open = True
        page.update()

    qr_dlg = ft.AlertDialog(
        title=ft.Text(
            spans=[
                ft.TextSpan(
                    "QR code",
                    ft.TextStyle(size=25)
                ),
                ft.TextSpan(
                    "\nVui lòng sử dụng camera để quét mã QR để truyền file",
                    ft.TextStyle(italic=True, size=20)
                ),
                ft.TextSpan(
                    "\n(Lưu ý: Cả hai thiết bị phải cùng truy cập vào cùng 1 Wifi)",
                    ft.TextStyle(italic=True,size=20)
                ),
            ]
            ),
        content= ft.Container(
            content=ft.Row(
                alignment=ft.MainAxisAlignment.CENTER,
                controls=[
                    ft.Column(
                        alignment=ft.MainAxisAlignment.CENTER,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            qr_info,
                        ]
                    ),
                ]
            )
        )
    )

    dlg_modal = ft.AlertDialog(
        modal=True,
        title=ft.Text(
            value = "Lưu ý",
            style = ft.TextStyle(size= 22),
            ),
        content=ft.Text(
            spans=[
                ft.TextSpan(
                    "Phần mềm sẽ nhận file thông qua mạng LAN ",
                    ft.TextStyle(size=20)
                ),
                ft.TextSpan(
                    "(Kết nối cùng wifi, không yêu cầu kết nối Internet)",
                    ft.TextStyle(italic=True, size=15)
                ),
                ft.TextSpan(
                    ", nhưng chưa tìm ra cách ngừng chạy localhost mà không tắt chương trình.",
                    ft.TextStyle(size=20)
                ),
                ft.TextSpan(
                    "\nBạn có muốn tiếp tục?",
                    ft.TextStyle(size=20)
                ),
                ft.TextSpan(
                    "\nBạn có thể truyền file qua các bên thứ ba như Google Drive, Zalo,...",
                    ft.TextStyle(italic=True, size=15)
                ),
            ]),
        actions=[
            ft.TextButton("Xác nhận", on_click=open_qr_modal),
            ft.TextButton("Hủy", on_click=close_dlg),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )

    #Route change
    def route_change(route):
        # page.views.clear()
        page.views.append(
            ft.View(
                route= "/home",
                vertical_alignment= ft.MainAxisAlignment.CENTER,
                horizontal_alignment= ft.MainAxisAlignment.CENTER,
                controls = [
                    ft.Container(
                        alignment=ft.alignment.center,
                        height = page.height,
                        width = page.width,
                        content=ft.Stack(
                            controls = [
                                ft.ElevatedButton(
                                text = "Lịch sử", 
                                top = 0,
                                right = 0,
                                on_click=lambda _: page.go("/history")),
                                ft.ElevatedButton(
                                text = "Sổ tay hướng dẫn sử dụng", 
                                top = 0,
                                left = 0,
                                on_click=lambda _: os.startfile(resource_path(r"views\notebook\notebook.html"))),
                                ft.Container(
                                    ft.Column(
                                            alignment= ft.MainAxisAlignment.CENTER,
                                            horizontal_alignment= ft.MainAxisAlignment.CENTER,
                                            controls = [
                                            ft.Container(
                                                margin=10,
                                                content = ft.Column(
                                                    controls = [
                                                        ft.ElevatedButton(
                                                            text = "Chọn file từ máy tính", 
                                                            width = 400,
                                                            on_click=lambda _: pick_files_dialog.pick_files(allow_multiple=False, allowed_extensions = ['jpg', 'png', 'JPG', 'PNG', 'jpeg', 'JPEG']),
                                                        ),
                                                        selected_files_path,
                                                        ]
                                                )),
                                            ft.Container(
                                                margin=10,
                                                content = ft.Column(
                                                    controls = [
                                                        ft.ElevatedButton(
                                                            text = "Nhận file từ thiết bị khác", 
                                                            width = 400,
                                                            on_click = lambda _: page.go("/recieve_file")),
                                                        qr_container,
                                                            ])
                                                            )
                                        ]),
                                    margin=40,
                                    alignment=ft.alignment.center
                                    ),
                                ]),
                        ),
            ]
            )
        )
        if ft.TemplateRoute(page.route).match("/history"):
            page.views.append(
                History_Screen(page)
            )
        elif ft.TemplateRoute(page.route).match("/file_to_doc/:path"): #and selected_files_path.value is not None:
            # print(ft.TemplateRoute(page.route).route)
            page.views.append(
                File_to_doc_Screen(page, re.compile(repath.pattern('/file_to_doc/:path')).match(ft.TemplateRoute(page.route).route).groupdict()['path'], detect_text_model, read_text_detector)
            )
        elif ft.TemplateRoute(page.route).match("/recieve_file"):
            image = open_dlg_modal_to_recieve_file_from_other_device()
            
        page.update()

    def view_pop(view):
        selected_files_path.value = ""
        selected_files_path.update()
        page.views.pop()
        top_view = page.views[-1]
        page.go(top_view.route)

    page.on_route_change = route_change
    page.on_view_pop = view_pop
    page.go(page.route)


ft.app(target=main)