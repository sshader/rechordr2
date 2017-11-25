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

		self.tempo_map  = SimpleTempoMap(60)
		self.sched = AudioScheduler(self.tempo_map)

		# connect scheduler into audio system
		self.audio.set_generator(self.sched)
		self.sched.set_generator(self.synth)
		self.last_tick = None
		self.song = []

		self.seq = NoteSequencer(self.sched, self.synth, 2, (0,40), self.song)

	def on_key_down(self, keycode, modifiers):
			if keycode[1] == 'p':
				self.seq.toggle()
				print self.song
			if keycode[1] == 'n':
				now = self.sched.get_tick()
				if self.last_tick != None:
					self.song.append((now - self.last_tick, 72))
				self.last_tick = now
					

	def on_update(self) :
				self.audio.on_update()


run(MainWidget)