import tkinter


# update to use this https://stackoverflow.com/a/49896477
class ButtonInfo:
    def __init__(self, button, msg, row, col):
        self.button = button
        self.msg = msg
        self.row = row
        self.col = col

class GUI:
    def __init__(self, size=23, offset_size=4):
        self.root = tkinter.Tk()
        self.root.title("OpCode Message generator")
        self.root.option_add("*font", 'Sans 15')
        self.default_font = ('Sans','15')
        self.bold_font = ('Sans','15','bold')

        self.size = size

        self.offset_size = offset_size

        self.rows = reversed([chr(i + ord('A')) for i in range(self.size)])
        self.cols = [chr(i + ord('A')) for i in range(self.size)]

        # title label
        self.title_area = tkinter.Frame(self.root)
        self.title = tkinter.Label(self.title_area, text="")
        self.title.pack()

        # button grid
        self.src_button_info = None
        self.dest_button_info = None

        self.button_area = tkinter.Frame(self.root, bg="#dfdfdf")

        self.msg_counter = 1

        self.button_grid = []
        for i, row in enumerate(self.rows):
            button_row = []
            for j, col in enumerate(self.cols):
                msg = f"{row}{col}"
                if all([offset_size <= i < size - offset_size, i % 2 == 0, offset_size <= j < size - offset_size, j % 2 == 0]):
                    color = 'yellow'
                else:
                    color = 'white'
                buttonij = tkinter.Button(self.button_area, bg=color, command=lambda tmp=msg, row=i, col=j: self.callable(tmp, row, col, self.button_grid))
                buttonij.grid(row=i, column=j)
                # buttonij.bind("<Enter>", self.on_enter)
                # buttonij.bind("<Leave>", self.on_leave)
                button_row.append(buttonij)
            self.button_grid.append(button_row)

        # status bar
        self.status_frame = tkinter.Frame(self.root)
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

        self.dropdown_area = tkinter.Frame(self.root, bg="#fdfdfd")

        self.variable = tkinter.StringVar(self.dropdown_area)
        self.variable.set(self.opcodes[0]) # default value
        self.variable.trace("w", self.dropdown_changed)
        
        self.dropdown = tkinter.OptionMenu(self.dropdown_area, self.variable, *self.opcodes)
        self.dropdown.pack()

        # show everything in a grid
        self.title_area.grid(row=0, column=0)
        self.button_area.grid(row=1, column=0, rowspan=size, sticky="nsew")
        self.dropdown_area.grid(row=size // 2 + 1, column=1, sticky="ew")
        self.status_frame.grid(row=size + 2, column=0, sticky="ew")

        self.dropdown_changed()

        self.root.mainloop()

    def dropdown_changed(self, *args):
        currop = self.variable.get()
        idx = self.opcodes.index(currop)
        if idx in (0, 1):
            text = "Select two squares; an opcode will be generated from the first to the second square.\n" \
                   "Note: the highlighted buttons represent the centers of chess squares."
        elif idx == 2:
            text = "Select a single square; an opcode will be generated to center the piece on that square."
        elif idx == 3:
            # TODO: implement displaying and selecting optypes
            text = "Select an optype."

        self.title.configure(text=text)
        # print("You changed the selection. The new selection is %s." % self.variable.get())

    def callable(self, msg, row, col, button_grid):
        # print(row, col)
        # print(msg)

        button = self.button_grid[row][col]
        if button.config('relief')[-1] == 'sunken':
            button.config(relief="raised", font=self.default_font)

            if self.dest_button_info is not None:
                self.dest_button_info = None
            else:
                self.src_button_info = None
        else:
            button.config(relief="sunken", font=self.bold_font)

            if self.src_button_info is None:
                self.src_button_info = ButtonInfo(button, msg, row, col)
            else:
                self.dest_button_info = ButtonInfo(button, msg, row, col)

        if self.src_button_info is not None and self.dest_button_info is not None:
            # create opcodemsg
            opcode = self.opcodes.index(self.variable.get())
            opcodemsg = f"~{opcode}{self.src_button_info.msg}{self.dest_button_info.msg}{self.msg_counter}"

            # update status bar
            self.status.configure(state='normal')
            self.status.delete(1.0, tkinter.END)
            self.status.insert(1.0, opcodemsg, "center")
            self.status.configure(state='disabled')

            # reset buttons
            self.src_button_info.button.config(relief="raised", font=self.default_font)
            self.src_button_info = None
            self.dest_button_info.button.config(relief="raised", font=self.default_font)
            self.dest_button_info = None

gui = GUI()
