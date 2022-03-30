import time
import tkinter
from tkinter import ttk

# update to use this https://stackoverflow.com/a/49896477
class ButtonInfo:
    def __init__(self, button, msg, row, col, color):
        self.button = button
        self.msg = msg
        self.row = row
        self.col = col
        self.orig_color = color


class GUI(tkinter.Tk):
    LARGEFONT =("Verdana", 35)

    BLUEHEX = "#FFFF00"
    YELLOWHEX = "#0000FF"
    WHITEHEX = "#FFFFFF"
    REDHEX = "#FF0000"

    DEFAULT_FONT = ('Sans', '15')
    BOLD_FONT = ('Sans', '15', 'bold')

    BOARDSIZE = 23
    OFFSETSIZE = 3

    def __init__(self, *args, **kwargs):
        # __init__ function for class Tk
        tkinter.Tk.__init__(self, *args, **kwargs)

        # creating a container
        container = tkinter.Frame(self) 
        container.pack(side = "top", fill = "both", expand = True)

        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)

        # Define board dimensions for subclasses
        self.rows = list(reversed([chr(i + ord('A')) for i in range(GUI.BOARDSIZE)]))
        self.cols = [chr(i + ord('A')) for i in range(GUI.BOARDSIZE)]

        # initializing frames to an empty array
        self.frames = {}

        # iterating through a tuple consisting
        # of the different page layouts
        for F in (OpCode0and1Page, OpCode2Page, OpCode3Page):
            frame = F(container, self)
            self.frames[F] = frame
  
            frame.grid(row = 0, column = 0, sticky ="nsew")

        self.msg_count = 0

        self.show_frame(OpCode0and1Page, 0)
        self.frames[OpCode0and1Page].dropdown_changed()

    # to display the current frame passed as
    # parameter
    def show_frame(self, cont, idx):
        frame = self.frames[cont]
        frame.previous_opcode_idx = idx
        frame.variable.set(frame.opcodes[idx])
        frame.tkraise()


class BaseFrame(tkinter.Frame):
    def __init__(self, parent, controller, title, b1text, b1frame, b2text, b2frame, board_size=23):
        tkinter.Frame.__init__(self, parent)

        self.controller = controller

        # title label
        self.title_area = tkinter.Frame(self)
        self.title = tkinter.Label(self.title_area, text=title)
        self.title.pack()

        # status bar
        self.status_frame = tkinter.Frame(self)
        self.status = tkinter.Text(self.status_frame, height=1, borderwidth=0)

        self.status.tag_configure("center", justify='center')
        self.status.insert(1.0, "Please select two squares...", "center")
        self.status.configure(state='disabled')
        self.status.pack()

        # button1 = ttk.Button(self, text=b1text, command=lambda: controller.show_frame(b1frame))

        # # putting the button in its place by using grid
        # button1.grid(row=1, column=1, padx=10, pady=10)

        # # button to show frame 2 with text layout2
        # button2 = ttk.Button(self, text=b2text, command=lambda: controller.show_frame(b2frame))

        # # putting the button in its place by using grid
        # button2.grid(row=2, column=1, padx=10, pady=10)

        # dropdown
        self.opcodes = [
            "0: MOVE_PIECE_IN_STRAIGHT_LINE",
            "1: MOVE_PIECE_ALONG_SQUARE_EDGES",
            "2: ALIGN_PIECE_ON_SQUARE",
            "3: INSTRUCTION"
        ]
        self.previous_opcode_idx = -1

        self.dropdown_area = tkinter.Frame(self, bg="#fdfdfd")

        self.variable = tkinter.StringVar(self.dropdown_area)
        self.variable.set(self.opcodes[0]) # default value
        self.variable.trace("w", self.dropdown_changed)
        
        self.dropdown = tkinter.OptionMenu(self.dropdown_area, self.variable, *self.opcodes)
        self.dropdown.pack()

        # show everything in a grid
        self.title_area.grid(row=0, column=0)
        self.dropdown_area.grid(row=0, column=1, sticky="ew")
        self.status_frame.grid(row=board_size + 2, column=0, sticky="ew")

    def dropdown_changed(self, *args):
        currop = self.variable.get()
        idx = self.opcodes.index(currop)
        if idx in (0, 1):
            text = "Select two squares; an opcode will be generated from the first to the second" \
                   " square.\nNote: the blue highlighted buttons represent centers of chess" \
                   " squares.\nThe yellow highlighted buttons represent corners of chess squares."
            frame = OpCode0and1Page
        elif idx == 2:
            text = "Select a single square; an opcode will be generated to center the piece on that square."
            frame = OpCode2Page
        elif idx == 3:
            # TODO: implement displaying and selecting optypes
            text = "Select an optype."
            frame = OpCode3Page

        self.previous_opcode_idx = idx

        self.title.configure(text=text)
        self.controller.show_frame(frame, idx)


class OpCode0and1Page(BaseFrame):
    def __init__(self, parent, controller):
        title = "Select two squares; an opcode will be generated from the first to the second" \
                " square.\nNote: the blue highlighted buttons represent centers of chess" \
                " squares.\nThe yellow highlighted buttons represent corners of chess squares."
        super().__init__(parent, controller, title, "Opcode 2 page", OpCode2Page, "Opcode 3 page", OpCode3Page)

        # title label
        self.title_area = tkinter.Frame(self)
        self.title = tkinter.Label(self.title_area, text="")
        self.title.pack()

        # button grid
        self.src_btn_info = None
        self.dest_btn_info = None

        self.button_area = tkinter.Frame(self, bg="#dfdfdf")

        self.button_grid = []
        for i, row in enumerate(self.controller.rows):
            button_row = []
            for j, col in enumerate(self.controller.cols):
                msg = f"{row}{col}"
                if GUI.OFFSETSIZE <= i < GUI.BOARDSIZE - GUI.OFFSETSIZE and \
                   GUI.OFFSETSIZE <= j < GUI.BOARDSIZE - GUI.OFFSETSIZE:
                    if i % 2 == 0 and j % 2 == 0:
                        color = GUI.YELLOWHEX
                    else:
                        color = GUI.BLUEHEX
                else:
                    color = GUI.WHITEHEX
                buttonij = tkinter.Button(
                    self.button_area, text="  o  ", highlightbackground=color,
                    command=lambda tmp=msg, row=i, col=j: self.callable(
                        tmp, row, col, self.button_grid))
                buttonij.grid(row=i, column=j)
                button_row.append(buttonij)
            self.button_grid.append(button_row)

        self.button_area.grid(row=1, column=0, rowspan=GUI.BOARDSIZE, sticky="nsew")

    def callable(self, msg, row, col, button_grid):
        # Clear buttons if two selected before selecting new button
        if self.src_btn_info is not None and self.dest_btn_info is not None:
            # reset buttons
            self.src_btn_info.button.config(relief="raised", font=GUI.DEFAULT_FONT,
                                            highlightbackground=self.src_btn_info.orig_color)
            self.src_btn_info = None

            self.dest_btn_info.button.config(relief="raised", font=GUI.DEFAULT_FONT,
                                             highlightbackground=self.dest_btn_info.orig_color)
            self.dest_btn_info = None

        button = self.button_grid[row][col]

        if self.src_btn_info is not None and button == self.src_btn_info.button:
            self.src_btn_info.button.config(relief="raised", font=GUI.DEFAULT_FONT,
                                            highlightbackground=self.src_btn_info.orig_color)
            self.src_btn_info = None
            return

        # If selecting a button that is already selected
        if button.config('relief')[-1] == 'sunken':
            button.config(relief="raised", font=GUI.DEFAULT_FONT)

            if self.dest_btn_info is not None:
                self.dest_btn_info = None
            else:
                self.src_btn_info = None
        # If no buttons already selected, store selected button as src, else dest
        else:
            if self.src_btn_info is None:
                self.src_btn_info = ButtonInfo(button, msg, row, col, button.cget("highlightbackground"))
            else:
                self.dest_btn_info = ButtonInfo(button, msg, row, col, button.cget("highlightbackground"))

            button.config(relief="sunken", font=GUI.BOLD_FONT, highlightbackground=GUI.REDHEX)

        # Compute opcode if two buttons selected
        if self.src_btn_info is not None and self.dest_btn_info is not None:
            # create opcodemsg
            opcode = self.opcodes.index(self.variable.get())
            opcodemsg = f"~{opcode}{self.src_btn_info.msg}{self.dest_btn_info.msg}{self.controller.msg_count}"
            self.controller.msg_count = (self.controller.msg_count + 1) % 2

            # update status bar
            self.status.configure(state='normal')
            self.status.delete(1.0, tkinter.END)
            self.status.insert(1.0, opcodemsg, "center")
            self.status.configure(state='disabled')


class OpCode2Page(BaseFrame):
    def __init__(self, parent, controller):
        title = "Select a single square; an opcode will be generated to center the piece on that "\
                "square."
        super().__init__(parent, controller, title, "Opcode 0/1 page", OpCode0and1Page, "Opcode 3 page", OpCode3Page)

        self.controller = controller

        # label = ttk.Label(self, text="this is opcode2page!", font = GUI.LARGEFONT)

        # # putting the grid in its place by using grid
        # label.grid(row=1, column=0, padx=10, pady=10)

        # button grid
        self.btn_info = None

        self.button_area = tkinter.Frame(self, bg="#dfdfdf")

        self.button_grid = []
        for i, row in enumerate(self.controller.rows):
            button_row = []
            for j, col in enumerate(self.controller.cols):
                msg = f"{row}{col}"
                if GUI.OFFSETSIZE <= i < GUI.BOARDSIZE - GUI.OFFSETSIZE and \
                   GUI.OFFSETSIZE <= j < GUI.BOARDSIZE - GUI.OFFSETSIZE:
                    if i % 2 == 0 and j % 2 == 0:
                        color = GUI.YELLOWHEX
                    else:
                        color = GUI.BLUEHEX
                else:
                    color = GUI.WHITEHEX
                buttonij = tkinter.Button(
                    self.button_area, text="  o  ", highlightbackground=color,
                    command=lambda tmp=msg, row=i, col=j: self.callable(
                        tmp, row, col, self.button_grid))
                buttonij.grid(row=i, column=j)
                button_row.append(buttonij)
            self.button_grid.append(button_row)

        self.button_area.grid(row=1, column=0, rowspan=GUI.BOARDSIZE, sticky="nsew")

    def callable(self, msg, row, col, button_grid):
        button = button_grid[row][col]

        if self.btn_info is not None:
            tmp = self.btn_info
            self.btn_info.button.config(relief="raised", font=GUI.DEFAULT_FONT,
                                            highlightbackground=self.btn_info.orig_color)
            self.btn_info = None
            if tmp.button == button:
                return

        # If selecting a button that is already selected
        if button.config('relief')[-1] == 'sunken':
            button.config(relief="raised", font=GUI.DEFAULT_FONT)
            self.btn_info = None
            return

        self.btn_info = ButtonInfo(button, msg, row, col, button.cget("highlightbackground"))
        button.config(relief="sunken", font=GUI.BOLD_FONT, highlightbackground=GUI.REDHEX)

        # create opcodemsg
        opcode = self.opcodes.index(self.variable.get())
        opcodemsg = f"~{opcode}{self.btn_info.msg}00{self.controller.msg_count}"
        self.controller.msg_count = (self.controller.msg_count + 1) % 2

        # update status bar
        self.status.configure(state='normal')
        self.status.delete(1.0, tkinter.END)
        self.status.insert(1.0, opcodemsg, "center")
        self.status.configure(state='disabled')

class OpCode3Page(BaseFrame):
    def __init__(self, parent, controller):
        title = "Select an optype."
        super().__init__(parent, controller, title, "Opcode 0/1 page", OpCode0and1Page, "Opcode 2 page", OpCode2Page)

        self.controller = controller

        label = ttk.Label(self, text="this is opcode3page!", font = GUI.LARGEFONT)

        # putting the grid in its place by using grid
        label.grid(row=1, column=0, padx=10, pady=10)

        # TODO: finish implementing here, just need a simple dropdown with instruction types.


gui = GUI()
gui.mainloop()
