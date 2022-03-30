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


class tkinterApp(tkinter.Tk):
    def __init__(self, *args, **kwargs):
        # __init__ function for class Tk
        tkinter.Tk.__init__(self, *args, **kwargs)

        # creating a container
        container = tkinter.Frame(self) 
        container.pack(side = "top", fill = "both", expand = True)

        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)

        # initializing frames to an empty array
        self.frames = {} 

        # iterating through a tuple consisting
        # of the different page layouts
        for F in (OpCode0and1Page, OpCode2Page, OpCode3Page):
            frame = F(container, self)
  
            # initializing frame of that object from
            # startpage, page1, page2 respectively with
            # for loop
            self.frames[F] = frame
  
            frame.grid(row = 0, column = 0, sticky ="nsew")
  
        self.show_frame(OpCode0and1Page)
  
    # to display the current frame passed as
    # parameter
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
  

class GUI(tkinter.Tk):
    LARGEFONT =("Verdana", 35)

    BLUEHEX = "#FFFF00"
    YELLOWHEX = "#0000FF"
    WHITEHEX = "#FFFFFF"
    REDHEX = "#FF0000"

    DEFAULT_FONT = ('Sans', '15')
    BOLD_FONT = ('Sans', '15', 'bold')

    def __init__(self, board_size=23, offset_size=3):
        # __init__ function for class Tk
        tkinter.Tk.__init__(self)

        self.title("OpCode Message generator")
        self.option_add("*font", 'Sans 15')
        # GUI.DEFAULT_FONT = ('Sans','15')
        # GUI.BOLD_FONT = ('Sans','15','bold')

        self.previous_opcode_idx = -1

        self.rows = reversed([chr(i + ord('A')) for i in range(board_size)])
        self.cols = [chr(i + ord('A')) for i in range(board_size)]

        # title label
        self.title_area = tkinter.Frame(self)
        self.title = tkinter.Label(self.title_area, text="")
        self.title.pack()

        # button grid
        self.src_btn_info = None
        self.dest_btn_info = None

        self.button_area = tkinter.Frame(self, bg="#dfdfdf")

        self.msg_counter = 1

        self.button_grid = []
        for i, row in enumerate(self.rows):
            button_row = []
            for j, col in enumerate(self.cols):
                msg = f"{row}{col}"
                if offset_size <= i < board_size - offset_size and \
                   offset_size <= j < board_size - offset_size:
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

        # status bar
        self.status_frame = tkinter.Frame(self)
        self.status = tkinter.Text(self.status_frame, height=1, borderwidth=0)

        self.status.tag_configure("center", justify='center')
        self.status.insert(1.0, "Please select two squares...", "center")
        self.status.configure(state='disabled')
        self.status.pack(fill="both", expand=True)

        # dropdown
        self.opcodes = [
            "MOVE_PIECE_IN_STRAIGHT_LINE",
            "MOVE_PIECE_ALONG_SQUARE_EDGES",
            "ALIGN_PIECE_ON_SQUARE",
            "INSTRUCTION"
        ]

        self.dropdown_area = tkinter.Frame(self, bg="#fdfdfd")

        self.variable = tkinter.StringVar(self.dropdown_area)
        self.variable.set(self.opcodes[0]) # default value
        self.variable.trace("w", self.dropdown_changed)
        
        self.dropdown = tkinter.OptionMenu(self.dropdown_area, self.variable, *self.opcodes)
        self.dropdown.pack()

        # show everything in a grid
        self.title_area.grid(row=0, column=0)
        self.button_area.grid(row=1, column=0, rowspan=board_size, sticky="nsew")
        self.dropdown_area.grid(row=board_size // 2 + 1, column=1, sticky="ew")
        self.status_frame.grid(row=board_size + 2, column=0, sticky="ew")

        self.dropdown_changed()

    def dropdown_changed(self, *args):
        currop = self.variable.get()
        idx = self.opcodes.index(currop)
        # Don't update GUI if we don't have to.
        if idx == self.previous_opcode_idx:
            return
        if idx in (0, 1):
            # Don't update GUI if we don't have to.
            if self.previous_opcode_idx in (0, 1):
                return
            text = "Select two squares; an opcode will be generated from the first to the second" \
                   " square.\nNote: the blue highlighted buttons represent centers of chess" \
                   " squares.\nThe yellow highlighted buttons represent corners of chess squares."
        elif idx == 2:
            text = "Select a single square; an opcode will be generated to center the piece on that square."
        elif idx == 3:
            # TODO: implement displaying and selecting optypes
            text = "Select an optype."

        self.previous_opcode_idx = idx

        self.title.configure(text=text)

    def callable(self, msg, row, col, button_grid):
        # print(row, col)
        # print(msg)

        button = self.button_grid[row][col]

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

        if self.src_btn_info is not None and self.dest_btn_info is not None:
            # create opcodemsg
            opcode = self.opcodes.index(self.variable.get())
            opcodemsg = f"~{opcode}{self.src_btn_info.msg}{self.dest_btn_info.msg}{self.msg_counter}"

            # update status bar
            self.status.configure(state='normal')
            self.status.delete(1.0, tkinter.END)
            self.status.insert(1.0, opcodemsg, "center")
            self.status.configure(state='disabled')

            # reset buttons
            self.src_btn_info.button.config(relief="raised", font=GUI.DEFAULT_FONT,
                                            highlightbackground=self.src_btn_info.orig_color)
            self.src_btn_info = None
            self.dest_btn_info.button.config(relief="raised", font=GUI.DEFAULT_FONT,
                                             highlightbackground=self.dest_btn_info.orig_color)
            self.dest_btn_info = None


class OpCode0and1Page(tkinter.Frame):
    def __init__(self, parent, controller):
        tkinter.Frame.__init__(self, parent)

        # label of frame Layout 2
        label = ttk.Label(self, text ="Opcode 0/1 page", font = GUI.LARGEFONT)

        # putting the grid in its place by using grid
        label.grid(row = 0, column = 4, padx = 10, pady = 10)

        button1 = ttk.Button(self, text ="Opcode 2 page",
        command = lambda : controller.show_frame(OpCode2Page))

        # putting the button in its place by using grid
        button1.grid(row = 1, column = 1, padx = 10, pady = 10)

        # button to show frame 2 with text layout2
        button2 = ttk.Button(self, text ="Opcode 3 page",
        command = lambda : controller.show_frame(OpCode3Page))

        # putting the button in its place by using grid
        button2.grid(row = 2, column = 1, padx = 10, pady = 10)


class OpCode2Page(tkinter.Frame):
    def __init__(self, parent, controller):
        tkinter.Frame.__init__(self, parent)

        # label of frame Layout 2
        label = ttk.Label(self, text ="Opcode 2 page", font = GUI.LARGEFONT)
         
        # putting the grid in its place by using grid
        label.grid(row = 0, column = 4, padx = 10, pady = 10)
  
        button1 = ttk.Button(self, text ="Opcode 0/1 page",
        command = lambda : controller.show_frame(OpCode0and1Page))

        # putting the button in its place by using grid
        button1.grid(row = 1, column = 1, padx = 10, pady = 10)

        # button to show frame 2 with text layout2
        button2 = ttk.Button(self, text ="Opcode 3 page",
        command = lambda : controller.show_frame(OpCode3Page))
     
        # putting the button in its place by using grid
        button2.grid(row = 2, column = 1, padx = 10, pady = 10)


class OpCode3Page(tkinter.Frame):
    def __init__(self, parent, controller):
        tkinter.Frame.__init__(self, parent)

        # label of frame Layout 2
        label = ttk.Label(self, text ="Opcode 3 page", font = GUI.LARGEFONT)
         
        # putting the grid in its place by using grid
        label.grid(row = 0, column = 4, padx = 10, pady = 10)
  
        button1 = ttk.Button(self, text ="Opcode 0/1 page",
        command = lambda : controller.show_frame(OpCode0and1Page))

        # putting the button in its place by using grid
        button1.grid(row = 1, column = 1, padx = 10, pady = 10)

        # button to show frame 2 with text layout2
        button2 = ttk.Button(self, text ="Opcode 2 page",
        command = lambda : controller.show_frame(OpCode2Page))
     
        # putting the button in its place by using grid
        button2.grid(row = 2, column = 1, padx = 10, pady = 10)


gui = GUI()
# gui = tkinterApp()
gui.mainloop()
