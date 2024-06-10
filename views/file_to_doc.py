import flet as ft
import os
from img2file_doc import get_image_to_doc
import uuid
import sys
import cv2
from lib.DocTr.app import process_image, init_Geotr_model
from PIL import Image

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def File_to_doc_Screen(page, path_to_image, detect_model, read_model):
    def pick_folder_result(e: ft.FilePickerResultEvent):
        selected_folder.value = e.path if e.path else ""
        selected_folder.update()
    pick_folder_path = ft.FilePicker(on_result=pick_folder_result)
    selected_folder = ft.Text(value="")
    page.overlay.append(pick_folder_path)

    file_name = ft.TextField(label="Tên file", width=200)

    loading_container = ft.Container(content=None)
    status_text = ft.Text(value = "Đang chuyển từ ảnh thành file Word...")
    status_icon_container = ft.Container(content=ft.ProgressRing())

    def confirm_image_to_doc(e):
        loading_container.content = ft.Row(
                                            alignment = ft.MainAxisAlignment.CENTER,
                                            vertical_alignment=ft.MainAxisAlignment.CENTER,
                                            controls = [
                                                status_icon_container, 
                                                status_text])
        info_saved_file_modal.open = False
        page.update()
        if GeoTr_checkbox.value:
            GeoTr_model = init_Geotr_model(resource_path(r'./lib/DocTr/model_pretrained/seg.pth'), resource_path(r'./lib/DocTr/model_pretrained/geotr.pth'))
            image = cv2.cvtColor(process_image(Image.open(path_to_image), GeoTr_model), cv2.COLOR_RGB2BGR)
            status = get_image_to_doc(detect_model, read_model, image, file_name.value, selected_folder.value)
        else:
            status = get_image_to_doc(detect_model, read_model, cv2.imread(path_to_image,1), file_name.value, selected_folder.value)
        if status == "IMAGE TO DOC SUCCESSFULLY":
            loading_container.content = None
            page.update()
            save_file_path.value = "Đọc thành công. Đường dẫn đến file là \n" + str(os.path.join(selected_folder.value, file_name.value +".docx"))
            save_file_path.update()
            open_saved_file.visible = True
            open_saved_file.update()
        else:
            status_text.value = status
            status_icon_container.content = ft.Icon(name=ft.icons.ERROR, color=ft.colors.RED)
            page.update()

    def close_info_saved_file_modal(e):
        info_saved_file_modal.open = False
        page.update()

    filename = ft.TextSpan(text= "Tên file: ")
    folderpath = ft.TextSpan(text="Thư mục chứa file: ")
    filepath = ft.TextSpan(text="Đường dẫn tới file: ")
    info_saved_file_modal = ft.AlertDialog(
        modal=True,
        title=ft.Text("Xác nhận thông tin"),
        content=ft.Text("File sẽ được lưu với các thông tin sau: ",
                        spans=[
                            filename,
                            folderpath,
                            filepath
                        ]),
        actions=[
            ft.TextButton("Yes", on_click=confirm_image_to_doc),
            ft.TextButton("No", on_click=close_info_saved_file_modal),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )

    def confirm_image_to_doc_modal(e):
        if file_name.value is None or file_name.value == "":
            file_name.value = uuid.uuid4().hex
        if selected_folder.value is None or selected_folder.value == "":
            if not os.path.isdir(resource_path(r'result')):
                os.makedirs(resource_path(r'result'), exist_ok=False)
            selected_folder.value = os.path.abspath(os.path.realpath(resource_path(r'result')))
        filename.text= "\nTên file: " + file_name.value + ".docx\n"
        folderpath.text="Thư mục chứa file: " + selected_folder.value + "\n"
        filepath.text="Đường dẫn tới file: " + selected_folder.value + "\\" + file_name.value + ".docx\n"
        page.dialog = info_saved_file_modal
        info_saved_file_modal.open = True
        page.update()

    def open_saved_file(e):
        os.startfile(str(os.path.join(selected_folder.value, file_name.value +".docx")))

    save_file_path = ft.Text(value="", max_lines= 30)
    open_saved_file =ft.ElevatedButton("Mở file", visible=False, on_click=open_saved_file)

    GeoTr_checkbox = ft.Checkbox(label="Làm phẳng mặt giấy", value=False)
    advanced_panel = ft.ExpansionPanelList(
        expand_icon_color=ft.colors.AMBER,
        elevation=8,
        divider_color=ft.colors.AMBER,
        controls=[
            ft.ExpansionPanel(
                can_tap_header = True,
                header=ft.ListTile(
                    title=ft.Text(
                        value="Cài đặt nâng cao",
                        style=ft.TextStyle(size=15)
                            )),
                    content=ft.Column(
                        controls = [
                                    GeoTr_checkbox,
                                    ]
                                ),
            )
        ]
    )

    return ft.View(
        route = "/file_to_doc",
        vertical_alignment= ft.MainAxisAlignment.CENTER,
        horizontal_alignment= ft.MainAxisAlignment.CENTER,
        controls = [
            ft.AppBar(title=ft.Text("Đọc ảnh thành file Word"), bgcolor=ft.colors.SURFACE_VARIANT),
            ft.Container(
                width = page.width,
                content = ft.ResponsiveRow(
                    alignment = ft.MainAxisAlignment.CENTER,
                    vertical_alignment= ft.MainAxisAlignment.CENTER,
                    width = page.width,
                    controls = [
                        ft.Column(
                            col={"md": 9}, controls=[
                            ft.Image(
                                src=f"{path_to_image}",
                                width=int(page.width*0.8)-100,
                                height=int(page.height*0.8),
                                fit=ft.ImageFit.CONTAIN,
                                repeat=ft.ImageRepeat.NO_REPEAT,
                                border_radius=ft.border_radius.all(10),
                            )]),
                        ft.Column(
                            col={"md": 3},
                            alignment = ft.MainAxisAlignment.CENTER,
                            horizontal_alignment= ft.MainAxisAlignment.CENTER,
                            controls = [
                                ft.Row(
                                    controls = [
                                    file_name,
                                    ft.Text(".docx")]
                                ),
                                ft.Column(
                                    controls = [
                                        selected_folder,
                                        ft.ElevatedButton(
                                            "Chọn thư mục để lưu",
                                            icon=ft.icons.FOLDER_ROUNDED,
                                            on_click=lambda _: pick_folder_path.get_directory_path(),
                                        ),
                                    ]
                                ),
                                advanced_panel,
                                ft.OutlinedButton(text="Xác nhận chuyển thành file", on_click=confirm_image_to_doc_modal),
                                ft.Column(
                                    controls = [
                                        loading_container,
                                        save_file_path,
                                        open_saved_file,
                                    ]
                                ),
                            ]
                        )
                    ]
                )
            )
        ],
    )