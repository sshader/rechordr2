import sys
sys.path.append('..')

from common.core import *
from common.audio import *
from common.synth import *
from common.clock import *
from common.metro import *
from common.noteseq import *

class MainWidget(BaseWidget):
	def __init__(self):
		super(MainWidget, self).__init__()

		self.audio = Audio(2)
		self.synth = Synth('data/FluidR3_GM.sf2')

		self.tempo_map  = SimpleTempoMap(30)
		self.sched = AudioScheduler(self.tempo_map)

		# connect scheduler into audio system
		self.audio.set_generator(self.sched)
		self.sched.set_generator(self.synth)
		self.song = [(480, 72), (480, 71), (240, 70), (240, 69), (480, 68)]

		self.seq = NoteSequencer(self.sched, self.synth, 2, (0,40), self.song)

	def on_key_down(self, keycode, modifiers):
			if keycode[1] == 'p':
				self.seq.toggle()

	def on_update(self) :
				self.audio.on_update()


run(MainWidget)