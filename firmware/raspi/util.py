class BoardCell:
	def __init__(self, x=None, y=None):
		''' Point, if no args passed, initialized to (0,0), top left corner of board
		
		Corresponds to center of board square
		'''
		self.x = x if x else 0
		self.y = y if y else 0

	def __str__(self):
		return f"{chr(self.x+ord('a'))}{chr(self.y+ord('a'))}"
