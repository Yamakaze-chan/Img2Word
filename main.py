import flet as ft
from views.history import History_Screen
from views.file_to_doc import File_to_doc_Screen
from views.receive_img import run_web_upload_image
from PIL import Image, ImageGrab
import base64
import io
import uuid
import win32clipboard
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
    page.theme_mode = "dark"
    page.theme = ft.theme.Theme(color_scheme_seed="blue")

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
        dlg_modal.open = False
        qr_dlg.open = False
        dlg_recheck_image_pasted.open = False
        page.update()

    def open_dlg_modal_to_recieve_file_from_other_device(e):
        if qr_text.value == None or qr_text.value == "":
            page.dialog = dlg_modal
            dlg_modal.open = True
        else:
            open_other_port_to_recieve_file_error_dlg = ft.AlertDialog(
                content=
                    ft.Text(
                        spans = [
                            ft.TextSpan("Phần mềm hiện đang tiếp nhận file thông qua đường dẫn\n"),
                            ft.TextSpan(qr_text.value.replace("Hoặc truy cập đường dẫn ", ""), style=ft.TextStyle(italic=True, size=20))
                            ],)
            )
            page.dialog = open_other_port_to_recieve_file_error_dlg
            open_other_port_to_recieve_file_error_dlg.open = True
        page.update()


    def change_theme_mode_btn(e):
        e.control.selected = not e.control.selected
        if page.theme_mode == "dark":
            page.theme = ft.theme.Theme(color_scheme_seed="green")
            page.theme_mode = "light"
        else:
            page.theme = ft.theme.Theme(color_scheme_seed="blue")
            page.theme_mode = "dark"
        e.control.update()
        page.update()

    pasted_image = ft.Image(
        width=500,
        height=500,
        fit=ft.ImageFit.CONTAIN,
        filter_quality=ft.FilterQuality.HIGH,
    )
    dlg_recheck_image_pasted = ft.AlertDialog(
        modal=True,
    )

    def image_from_clipboard_to_doc(e):
        if not os.path.isdir(resource_path(r'upload_img')):
            os.makedirs(resource_path(r'upload_img'), exist_ok=False)
        pasted_image_path = os.path.join(resource_path(r"upload_img"),(str(uuid.uuid4().hex)+".png"))
        print(pasted_image_path)
        Image.open(io.BytesIO(base64.b64decode(pasted_image.src_base64))).save(pasted_image_path)
        page.go(route="/file_to_doc"+"/"+pasted_image_path)

    def recheck_image_paste_from_clipboard():
        try:
            img_byte_arr = io.BytesIO()
            ImageGrab.grabclipboard().save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            pasted_image.src_base64 = base64.b64encode(img_byte_arr).decode("utf-8")
            dlg_recheck_image_pasted.content=ft.Column(
                controls=[
                    ft.Text(value="Bạn có muốn chuyển ảnh dưới đây thành Word?"),
                    pasted_image,
                ]
            )
            dlg_recheck_image_pasted.actions=[
                ft.TextButton("Xác nhận", on_click=image_from_clipboard_to_doc),
                ft.TextButton("Hủy", on_click=close_dlg),
            ]
        except Exception as e:
            win32clipboard.OpenClipboard()
            dlg_recheck_image_pasted.content = ft.Column(
                controls=[
                        ft.Text("Dữ liệu bạn mới sao chép hình như không phải định dạng ảnh, phiền bạn kiểm tra lại"),
                        ft.Text("Dữ liệu mà phần mềm đọc được là: "),
                        ft.Text(win32clipboard.GetClipboardData())
                ]
                )
            win32clipboard.CloseClipboard()
            dlg_recheck_image_pasted.actions=[
                ft.TextButton("Để tôi kiểm tra lại", on_click=close_dlg),
            ]
        page.dialog = dlg_recheck_image_pasted
        dlg_recheck_image_pasted.open = True
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
                                ft.IconButton(
                                icon=ft.icons.DARK_MODE_OUTLINED,
                                selected_icon=ft.icons.WB_SUNNY_OUTLINED,
                                on_click=change_theme_mode_btn,
                                bottom = 10,
                                right = 0,
                                selected=False),
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
                                                            on_click = open_dlg_modal_to_recieve_file_from_other_device),
                                                        qr_container,
                                                            ])
                                                            ),
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
        # elif ft.TemplateRoute(page.route).match("/recieve_file"):
        #     image = open_dlg_modal_to_recieve_file_from_other_device()
            
        page.update()

    def view_pop(view):
        selected_files_path.value = ""
        selected_files_path.update()
        page.views.pop()
        top_view = page.views[-1]
        page.go(top_view.route)

    page.on_route_change = route_change
    page.on_view_pop = view_pop
    def on_keyboard(e: ft.KeyboardEvent):
            on_keyboard_input_data = e.data.replace('{', '', 1).replace('}', '', -1).replace("\"",'').split(',')
            on_keyboard_input_data = {i.split(':')[0]: i.split(':')[1] for i in on_keyboard_input_data}
            if e.ctrl and on_keyboard_input_data["key"]=="V":
                recheck_image_paste_from_clipboard()

    page.on_keyboard_event = on_keyboard
    page.go(page.route)


ft.app(target=main)