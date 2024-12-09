import time
import argparse

import numpy as np
np.bool = np.bool_

from PyQt5 import QtWidgets, QtCore

from vispy.scene import SceneCanvas, visuals
from vispy.app import use_app

from scipy.fft import fft, ifft,fftfreq

from scipy.ndimage import gaussian_filter

# Define the parser
parser = argparse.ArgumentParser(description='Mixing Simulator')
parser.add_argument('-n', action="store", dest='Np', default=2**8)
parser.add_argument('-p', action="store", dest='p', default=3)
args = parser.parse_args()

Np=int(args.Np)
p=int(args.p)

tmax=2000
version=' v1.0 06/2024'

IMAGE_SHAPE = (Np, Np*p)  # (height, width)
CANVAS_SIZE = (Np*p*3,Np*3)  # (width, height)

COLORMAP_CHOICES = ["viridis", 'binary', 'gist_gray', 'plasma', 'inferno', 'magma', 'cividis',"reds", "blues"]
SIMULATION_CHOICES = ["Gradient","Decay","Source"]
IMAGE_CHOICES = ["Scalar","Vx","Vy","Vnorm"]


class Controls(QtWidgets.QWidget):
	def __init__(self, parent=None):
		super().__init__(parent)
		
		# Sine flow parameters
		self.power = -5/3.
		self.lmin = 0.01
		self.lmax= 1.0
		self.a = 0.75
		self.D=-5
		self.mode='Source'
		self.speed=0.05
		self.s=int(Np*0.05)
		self.tcorr=20
		self.fps=60
		self.mode="Gradient"
		self.start=True
		self.quit=False
		self.imtype="Scalar"
		
		layout = QtWidgets.QVBoxLayout()
		
		self.title =  QtWidgets.QLabel("\n  Mixing Simulator \n "+version +"\n\n")
		layout.addWidget(self.title)
		
		
		self.start_bt = QtWidgets.QPushButton('Start/Pause', self)
		layout.addWidget(self.start_bt)
		
# 		self.quit_bt = QtWidgets.QPushButton('Quit', self)
# 		layout.addWidget(self.quit_bt)
		
# 		self.pause_chooser = QtWidgets.QComboBox()
# 		self.pause_chooser.addItems(SIMULATION_CHOICES)
# 		layout.addWidget(self.pause_chooser)
		
		self.mode_label = QtWidgets.QLabel("Simulation mode:")
		layout.addWidget(self.mode_label)
		self.mode_chooser = QtWidgets.QComboBox()
		self.mode_chooser.addItems(SIMULATION_CHOICES)
		layout.addWidget(self.mode_chooser)
		
		
		self.imtype_label = QtWidgets.QLabel("Plot:")
		layout.addWidget(self.imtype_label)
		self.imtype_chooser = QtWidgets.QComboBox()
		self.imtype_chooser.addItems(IMAGE_CHOICES)
		layout.addWidget(self.imtype_chooser)
		
		self.colormap_label = QtWidgets.QLabel("Colormap:")
		layout.addWidget(self.colormap_label)
		self.colormap_chooser = QtWidgets.QComboBox()
		self.colormap_chooser.addItems(COLORMAP_CHOICES)
		layout.addWidget(self.colormap_chooser)
		
		self.rescale_bt = QtWidgets.QPushButton('Rescale', self)
		layout.addWidget(self.rescale_bt)
		# Slider
		#layout2=QtWidgets.QHBoxLayout()
		# power (x100)
		
		# Speed slider
		self.fps_label = QtWidgets.QLabel("FPS: {:1.1f}".format(self.fps))
		layout.addWidget(self.fps_label)
		self.fps_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
		self.fps_sl.setMinimum(1)
		self.fps_sl.setMaximum(1200)
		self.fps_sl.setValue(600)
		self.fps_sl.setTickPosition(QtWidgets.QSlider.TicksAbove)
		self.fps_sl.setTickInterval(100)
		layout.addWidget(self.fps_sl)
		
		# Roughness slider
		self.power_label = QtWidgets.QLabel("Rougness: {:1.2f}".format(self.power))
		layout.addWidget(self.power_label)
		self.power_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
		self.power_sl.setMinimum(-500)
		self.power_sl.setMaximum(0)
		self.power_sl.setValue(-83)
		self.power_sl.setTickPosition(QtWidgets.QSlider.TicksAbove)
		self.power_sl.setTickInterval(100)
		layout.addWidget(self.power_sl)
		
		
		self.D_label = QtWidgets.QLabel("Diffusion (log): {:1.1f}".format(self.D))
		layout.addWidget(self.D_label)
		self.D_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
		self.D_sl.setMinimum(int(np.floor(np.log10((0.2/Np**2))))*10)
		self.D_sl.setMaximum(0)
		self.D_sl.setValue(-50)
		self.D_sl.setTickPosition(QtWidgets.QSlider.TicksAbove)
		self.D_sl.setTickInterval(10)
		layout.addWidget(self.D_sl)

		# Amplitude slider
		self.a_label = QtWidgets.QLabel("Amplitude: {:1.2f}".format(self.a))
		layout.addWidget(self.a_label)
		self.a_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
		self.a_sl.setMinimum(0)
		self.a_sl.setMaximum(50)
		self.a_sl.setValue(5)
		self.a_sl.setTickPosition(QtWidgets.QSlider.TicksAbove)
		self.a_sl.setTickInterval(10)
		layout.addWidget(self.a_sl)
		
		# Correlation time
		self.tcorr_label = QtWidgets.QLabel("Corr. time: {:1.2f}".format(self.tcorr))
		layout.addWidget(self.tcorr_label)
		self.tcorr_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
		self.tcorr_sl.setMinimum(0)
		self.tcorr_sl.setMaximum(50)
		self.tcorr_sl.setValue(5)
		self.tcorr_sl.setTickPosition(QtWidgets.QSlider.TicksAbove)
		self.tcorr_sl.setTickInterval(10)
		layout.addWidget(self.tcorr_sl)
		
		# Max lengthscale
		self.lmax_label = QtWidgets.QLabel("Max Lengthscales: {:1.2f}".format(self.lmax))
		layout.addWidget(self.lmax_label)
		self.lmax_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
		self.lmax_sl.setMinimum(0)
		self.lmax_sl.setMaximum(100)
		self.lmax_sl.setValue(100)
		self.lmax_sl.setTickPosition(QtWidgets.QSlider.TicksAbove)
		self.lmax_sl.setTickInterval(20)
		layout.addWidget(self.lmax_sl)

		# Min lengthscale
		self.lmin_label = QtWidgets.QLabel("Min Lengthscales: {:1.2f}".format(self.lmin))
		layout.addWidget(self.lmin_label)
		self.lmin_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
		self.lmin_sl.setMinimum(1)
		self.lmin_sl.setMaximum(100)
		self.lmin_sl.setValue(1)
		self.lmin_sl.setTickPosition(QtWidgets.QSlider.TicksAbove)
		self.lmin_sl.setTickInterval(20)
		layout.addWidget(self.lmin_sl)

		#layout.SetMaximumSize
		
		layout.addStretch(1)
		self.setLayout(layout)

	def set_amplitude(self,r):
		self.a=r/10
		self.a_label.setText("Amplitude: {:1.1f}".format(self.a))

	def set_diffusion(self,r):
		self.D=r/10
		self.D_label.setText("Diffusion (log): {:1.2f}".format(self.D))
	
	def set_roughness(self,r):
		self.power=r/100
		self.power_label.setText("Rougness: {:1.2f}".format(self.power))
		
	def set_tcorr(self,r):
		self.tcorr=r
		self.tcorr_label.setText("Corr time: {:1.2f}".format(self.tcorr))
		
	def set_lmax(self,r):
		self.lmax=r/100
		self.lmax_label.setText("Max Lengthscale: {:1.2f}".format(self.lmax))
		
	def set_lmin(self,r):
		self.lmin=r/100
		self.lmin_label.setText("Min Lengthscale: {:1.2f}".format(self.lmin))
		
	def set_fps(self,r):
		self.fps=r/10
		self.fps_label.setText("FPS: {:1.1f}".format(self.fps))

	def set_mode(self, _mode: str):
		self.mode = _mode

	def set_start(self):
		self.start=not(self.start)

	def set_quit(self):
		self.quit=True
		
	def set_imtype(self, imt: str):
		self.imtype = imt
		
class CanvasWrapper:
	def __init__(self):
		
		self.canvas = SceneCanvas(size=CANVAS_SIZE)
#		self.grid = self.canvas.central_widget.add_grid()
	
		self.view_top = self.canvas.central_widget.add_view()

#		self.view_top = self.grid.add_view(0, 0, bgcolor='cyan')
		image_data = np.zeros(IMAGE_SHAPE)
		
		
		self.image = visuals.Image(
			image_data,
			texture_format="auto",
			clim=[-1,1],
			cmap=COLORMAP_CHOICES[0],
			parent=self.view_top.scene,
			interpolation='bilinear'
		)
		
		#self.view_top.camera.PanZoomCamera(parent=self.view_top.scene, aspect=1, name='panzoom')
		self.view_top.camera = "panzoom"
		#self.view_top.camera = cameras.base_camera.BaseCamera(aspect=1,interactive=False)
		self.view_top.camera.set_range(x=(0, IMAGE_SHAPE[1]), y=(0, IMAGE_SHAPE[0]), margin=0)
		self.view_top.camera.interactive=False
		
		# Point source on mouse click
		self.blob=[]
		self.canvas.events.mouse_release.connect(self.set_blob)


# 		self.view_bot = self.grid.add_view(1, 0, bgcolor='#c0c0c0')
# 		line_data = _generate_random_line_positions(NUM_LINE_POINTS)
# 		self.line = visuals.Line(line_data, parent=self.view_bot.scene, color=LINE_COLOR_CHOICES[0])
# 		self.view_bot.camera = "panzoom"
# 		self.view_bot.camera.set_range(x=(0, NUM_LINE_POINTS), y=(0, 1))

	def set_blob(self,event):
		if event.button == 1:
			# left click
			transform = self.image.transforms.get_transform(map_to="canvas")
			img_x, img_y = transform.imap(event.pos)[:2]
			# optionally do the below to tell other handlers not to look at this event:
			#event.handled = True
			self.blob=[img_x,img_y,1]
		if event.button == 2:
			# left click
			transform = self.image.transforms.get_transform(map_to="canvas")
			img_x, img_y = transform.imap(event.pos)[:2]
			# optionally do the below to tell other handlers not to look at this event:
			#event.handled = True
			self.blob=[img_x,img_y,-1]

	
	def set_image_colormap(self, cmap_name: str):
		print(f"Changing image colormap to {cmap_name}")
		self.image.cmap = cmap_name

		
# 	def set_line_color(self, color):
# 		print(f"Changing line color to {color}")
# 		self.line.set_data(color=color)

	def update_data(self, new_data_dict):
		#print("Updating data...")
#		self.line.set_data(new_data_dict["line"])
		self.image.set_data(new_data_dict["image"])
		self.canvas.update()

class MyMainWindow(QtWidgets.QMainWindow):
	closing = QtCore.pyqtSignal()

	def __init__(self, canvas_wrapper: CanvasWrapper, *args, **kwargs):
		super().__init__(*args, **kwargs)

		central_widget = QtWidgets.QWidget()
		main_layout = QtWidgets.QHBoxLayout()

		self._controls = Controls()
		main_layout.addWidget(self._controls)
		self._canvas_wrapper = canvas_wrapper
		main_layout.addWidget(self._canvas_wrapper.canvas.native)

		central_widget.setLayout(main_layout)
		self.setCentralWidget(central_widget)

		self._connect_controls()

	def _connect_controls(self):
		self._controls.mode_chooser.currentTextChanged.connect(self._controls.set_mode)
		self._controls.colormap_chooser.currentTextChanged.connect(self._canvas_wrapper.set_image_colormap)
		self._controls.imtype_chooser.currentTextChanged.connect(self._controls.set_imtype)
#		self._controls.line_color_chooser.currentTextChanged.connect(self._canvas_wrapper.set_line_color)
		self._controls.power_sl.valueChanged.connect(self._controls.set_roughness)
		self._controls.D_sl.valueChanged.connect(self._controls.set_diffusion)
		self._controls.a_sl.valueChanged.connect(self._controls.set_amplitude)
		self._controls.tcorr_sl.valueChanged.connect(self._controls.set_tcorr)
		self._controls.lmin_sl.valueChanged.connect(self._controls.set_lmin)
		self._controls.lmax_sl.valueChanged.connect(self._controls.set_lmax)
		self._controls.fps_sl.valueChanged.connect(self._controls.set_fps)
		self._controls.start_bt.clicked.connect(self._controls.set_start)
		#self._controls.quit_bt.clicked.connect(self._controls.set_quit)
		self._controls.rescale_bt.clicked.connect(self.set_rescale)

	def set_rescale(self):
		C=np.array(self._canvas_wrapper.image._data)
		self._canvas_wrapper.image.clim=[np.min(C),np.max(C)]
	
	def closeEvent(self, event):
		print("Closing main window!")
		self.closing.emit()
		return super().closeEvent(event)


class DataSource(QtCore.QObject):
	"""Object representing a complex data producer."""
	new_data = QtCore.pyqtSignal(dict)
	finished = QtCore.pyqtSignal()

	def __init__(self, myMainWindow: MyMainWindow, parent=None):
		super().__init__(parent)
		self._should_end = False
		self._image_data = np.zeros(IMAGE_SHAPE)
#		self._line_data = _generate_random_line_positions(NUM_LINE_POINTS)
		self._myMainWindow = myMainWindow
		self._tmax=1000
		self._D=10**self._myMainWindow._controls.D
		self._power=self._myMainWindow._controls.power
		self._lmin=self._myMainWindow._controls.lmin
		self._lmax=self._myMainWindow._controls.lmax
		self._a=self._myMainWindow._controls.a
		self._mode=self._myMainWindow._controls.mode
		self._s=self._myMainWindow._controls.s
		self._tcorr=self._myMainWindow._controls.tcorr
		self._fps=self._myMainWindow._controls.fps
		self._imtype=self._myMainWindow._controls.imtype
		self._quit=self._myMainWindow._controls.quit
		self._flow='1D'
		
	def run_data_creation(self):
		# local parameters
		
		#print(Np*p,self._tmax,self._tcorr)
		
		if self._flow=='1D':
			VFX=self.noise_corr(Np*p,self._tmax,self._tcorr)
			VFY=self.noise_corr(Np,self._tmax,self._tcorr)
		else:
			PhiFf=self.noise_corr_phi(self._tmax,self._tcorr)

		C=np.zeros((Np,p*Np))
		X,Y=np.meshgrid(np.arange(Np),np.arange(p*Np))
		
		ky=2*np.pi*np.tile(fftfreq(C.shape[0], d=1.0/C.shape[0]),(C.shape[1],1)).T
		kx=2*np.pi*np.tile(fftfreq(C.shape[1], d=1.0/C.shape[0]),(C.shape[0],1))
	
			
		k=np.sqrt(ky**2+kx**2)
		
		# constant terms
		xs=[0.95*p,0.5]
		mass=100
		B=mass*np.exp(-1j*kx*xs[0])*np.exp(-1j*ky*xs[1]) # Dirac Point source in fourirer
		B=B*np.exp(-1e-4*k**2)
		
		# Start from the fourier transform of concentration field
		t=0
		#vXold=a*np.tile(flow(),(Np,1)).T
		#vYold=a*np.tile(flow(),(Np,1))	
		fC=fft(fft(C,axis=0),axis=1)
		
		
#		while not(self._quit): #main loop
		while True: #main loop
			# If start button is pressed 
			while not(self._myMainWindow._controls.start):
				time.sleep(1)
				
			# local parameters
			self._D=10**self._myMainWindow._controls.D
			self._power=self._myMainWindow._controls.power
			self._lmax=self._myMainWindow._controls.lmax
			
			if self._myMainWindow._controls.lmin>=self._lmax*0.9:
				self._myMainWindow._controls.lmin_sl.setValue(1)
				self._lmin=0.01
			else:				
				self._lmin=self._myMainWindow._controls.lmin
				
			self._a=self._myMainWindow._controls.a
			self._mode=self._myMainWindow._controls.mode
			self._s=self._myMainWindow._controls.s
			self._fps=self._myMainWindow._controls.fps
			self._imtype=self._myMainWindow._controls.imtype
			self._quit=self._myMainWindow._controls.quit
			
			# If correlation time change, regenerate velocity field
			if self._tcorr!=self._myMainWindow._controls.tcorr:
				self._tcorr=self._myMainWindow._controls.tcorr
				if self._flow=='1D':
					VFX=self.noise_corr(Np*p,self._tmax,self._tcorr)
					VFY=self.noise_corr(Np,self._tmax,self._tcorr)
				else:
					PhiFf=self.noise_corr_phi(self._tmax,self._tcorr)
			
			#self._flow=self._myMainWindow._controls.flow
			
			dt=0.1
			A=np.exp(-self._D*k**2*dt) # Diffusion
			
			
			# Solve advection diffusion on frequency doamin
			t=t+1
			ti=time.time()
			
			if self._flow=='1D':
				if self._mode=='Source':
					vX=self._a/self._tcorr*np.tile(self.flow_1D(VFX[:,np.mod(t,self._tmax)],flow=True,t=t),(Np,1))
				else:
					vX=self._a/self._tcorr*np.tile(self.flow_1D(VFX[:,np.mod(t,self._tmax)]),(Np,1))
					
				vY=self._a/self._tcorr*np.tile(self.flow_1D(VFY[:,np.mod(t,self._tmax)]),(Np*p,1)).T
				if self._mode=='Source':
					vY=vY+self._a/self._tcorr*2.0 # mean flow
			
			if self._flow=='2D':
				vX,vY=self.flow_2D(PhiFf[np.mod(t,self._tmax),:,:])
				vY=self._a/self._tcorr*vY/10
				vX=self._a/self._tcorr*vX/10
				
			#print(vX.shape,vY.shape)
			
		
			if self._flow=='1D':
				fCx = ifft(fC,axis=0)
				
				
				dcx=np.exp(1j*kx*vY*dt)*fCx
				
				if self._mode=='Gradient':
					dcx[kx==0]=fCx[kx==0]-1.0*vY[kx==0]*dt*Np*p
				
				fC=fft(dcx,axis=0)*A#			# With source term
				if self._mode=='Source':
					fC=fC+B*dt
					
				fCy = ifft(fC,axis=1)
				fC=fft(np.exp(1j*ky*vX*dt)*fCy,axis=1)*A
				if self._mode=='Source':
					fC=fC+B*dt
				
# 				
			if self._flow=='2D':
				# Need to use a Adams-Bashforth time stepping + 2/3 deliasing rule
				fC=fC*np.exp(1j*ky*vY*dt+1j*kx*vX*dt)*A
				fC[k>3/2*Np-2]=0 # dealiasing
				print(max(k),3/2*Np-2)

			C=np.real(ifft(ifft(fC,axis=0),axis=1))
			
			if self._mode=='Source': # Implement no periodicity in source mode
				nb=10 # should be larger than CFL
				C[:,:nb]=0
				C[:,-nb:]=0		
				C[:nb,:]=0
				C[-nb:,:]=0		
				if self._mode!='Gradient':
					C=C-C.mean()
				fC=fft(fft(C,axis=0),axis=1)
				
				#Cw=np.uint8((C-np.min(C))/(np.max(C)-np.min(C))*255)
				#cv2.imwrite('/data/video/img{:06d}.jpg'.format(t),Cw)
		
			if self._imtype=="Scalar":
				data_dict = { "image": C }
			if self._imtype=="Vx":
				data_dict = { "image": vX }
			if self._imtype=="Vy":
				data_dict = { "image": vY }
			if self._imtype=="Vnorm":
				data_dict = { "image": np.sqrt(vX**2+vY**2) }
				
			self.new_data.emit(data_dict)
			
			if len(self._myMainWindow._canvas_wrapper.blob)>0:
				s=20
				x=int(self._myMainWindow._canvas_wrapper.blob[0])
				y=int(self._myMainWindow._canvas_wrapper.blob[1])
				c0=self._myMainWindow._canvas_wrapper.blob[2]
				C[y-s:y+s,x-s:x+s]=c0#np.max(C)
				self._myMainWindow._canvas_wrapper.blob=[]
				fC=fft(fft(C,axis=0),axis=1)
			
			elapsed=time.time()-ti
			print('Max fps :',1/elapsed)
			#print(elapsed)
			if elapsed < 1./self._fps:
				time.sleep( 1./self._fps-elapsed)
 	
		print("Exiting ...")
		self.stop_data()
		#self.finished.emit()


	def stop_data(self):
		print("Data source is quitting...")
		self._should_end = True

	def noise_corr(self,n,tmax,tcorr):
		V=np.random.randn(n,tmax)
		VF=fft(V,axis=0)
		k=2*np.pi*fftfreq(n, d=1.0/Np)
		VFf=[]
		for i in range(VF.shape[0]):
			tf=np.uint16(np.minimum(np.maximum(1/k[i]/2/np.pi*Np*tcorr,1),Np))
			filt=gaussian_filter(VF[i,:],tf,mode='wrap')
# 			win = signal.windows.hann(tf)
# 			filt=signal.convolve(VF[i,:], win, mode='wrap') / sum(win)
			VFf.append(filt/np.mean(np.abs(filt))*np.mean(np.abs(VF[i,:])))
		VFf=np.array(VFf)
		VFf[np.isnan(VFf)]=0
		return VFf
	
	def noise_corr_phi(self,tmax,tcorr):
		Phi=np.random.randn(tmax,Np,Np*p)
		PhiF=fft(fft(Phi,axis=1),axis=2)
		ky=2*np.pi*np.tile(fftfreq(Np, d=1.0/Np),(Np*p,1)).T
		kx=2*np.pi*np.tile(fftfreq(Np*p, d=1.0/Np),(Np,1))
		k=np.sqrt(ky**2+kx**2)
		PhiFf=np.copy(PhiF)
		for i in range(PhiF.shape[1]):
			for j  in range(PhiF.shape[2]):
				tf=np.uint16(np.minimum(np.maximum(1/k[i,j]/2/np.pi*Np*tcorr,1),Np))
				filt=gaussian_filter(PhiFf[:,i,j],tf,mode='wrap')
				PhiFf[:,i,j]=filt
		return PhiFf


	def flow_1D(self,VF,flow=True,t=0):
		k=2*np.pi*fftfreq(len(VF), d=1.0/Np)
		K=k**(self._power*0.5)
		K[k<np.pi*2/self._lmax]=0
		K[k>np.pi*2/self._lmin]=0
		VFK=(VF.T*K).T
		VFK[np.isnan(VFK)]=0
		if self._mode=='Source': #shift by mean flow
			VFK=VFK*np.exp(1j*t*self._a/self._tcorr*2.0)
		v=np.real(np.fft.ifft(VFK,axis=0))
		return v/np.std(v)

	def flow_2D(self,PhiFf,flow=True,t=0):
		ky=2*np.pi*np.tile(fftfreq(Np, d=1.0/Np),(Np*p,1)).T
		kx=2*np.pi*np.tile(fftfreq(Np*p, d=1.0/Np),(Np,1))
		k=np.sqrt(ky**2+kx**2)
		K=k**(self._power-1)# -1 because it is the streamfunction( need to derive to get velocity)
		K[k<np.pi*2/self._lmax]=0
		K[k>np.pi*2/self._lmin]=0
		PhiFfk=PhiFf*K
		PhiFfk[np.isnan(PhiFfk)]=0
		vxf=PhiFfk*np.exp(-1j*ky)
		vyf=-PhiFfk*np.exp(-1j*kx)
		vx=np.real(ifft(ifft(vxf,axis=1),axis=0))
		vy=np.real(ifft(ifft(vyf,axis=1),axis=0))
		vn=np.sqrt(vx**2+vy**2)
		return vx/np.std(vn),vy/np.std(vn)

if __name__ == "__main__":
	app = use_app("pyqt5")
	app.create()

	canvas_wrapper = CanvasWrapper()
	win = MyMainWindow(canvas_wrapper)
	data_thread = QtCore.QThread(parent=win)
	data_source = DataSource(win)
	data_source.moveToThread(data_thread)

	# update the visualization when there is new data
	data_source.new_data.connect(canvas_wrapper.update_data)
	
	# start data generation when the thread is started
	data_thread.started.connect(data_source.run_data_creation)
	
	# if the data source finishes before the window is closed, kill the thread
	data_source.finished.connect(data_thread.quit, QtCore.Qt.DirectConnection)
	
	# if the window is closed, tell the data source to stop
	win.closing.connect(data_source.stop_data, QtCore.Qt.DirectConnection)
	
	# when the thread has ended, delete the data source from memory
	data_thread.finished.connect(data_source.deleteLater)

	win.show()
	data_thread.start()
	app.run()

	print("Waiting for data source to close gracefully...")
	data_thread.wait(5000)
