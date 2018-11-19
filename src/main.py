import pandas as pd
import numpy as np

from bokeh.io import show

from bokeh.layouts import row, widgetbox, column, gridplot
from bokeh.models import Select
from bokeh.palettes import Spectral5
from bokeh.plotting import curdoc, figure
from bokeh.sampledata.autompg import autompg_clean as df
from bokeh.embed import components

from bokeh.models.widgets import TextInput
from bokeh.models import LogColorMapper, LogTicker, ColorBar
from bokeh.models import ColumnDataSource, OpenURL, TapTool
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.models.widgets import Div

from bokeh.models.widgets import PreText

from collections import OrderedDict
from ulirg_fitting import counts_out
from astropy.io import fits

COLORS = Spectral5
N_COLORS = len(COLORS)
format_all = '{:.2e}'.format


ulirg_num = ["1","2","3","4","5"]
ulirg_names = ["1","2","3","4","5"]


filters = ['F125', 'F140', 'F150', 'F165']
data_dir  = Path('./data/')


params = ['ulirg', 'x_plot', 'y_plot', 'A','b', 'm',  'A_err', 'b_err', 'm_err']


import pysynphot as S


band_names = ['f125lp', 'f140lp', 'f150lp', 'f165lp']

bands = [S.ObsBandpass('acs,sbc,%s' % band) for band in band_names]
waves = [band.wave for band in bands]

from bokeh.models.widgets import Button

s
button = Button(label="ULIRG fitting method for LYA", button_type ='success')

pre = PreText(text="""details are at 

    http://homepages.spa.umn.edu/~sourabh/projects/docs/build/html/index.html?""",
width=300, height=100)


def plot_lya(data_dir, filter_name, x_range, y_range):
    
    data = fits.getdata(data_dir+'ULIRG%s/gal%s_UV_%s_x550_y550_sz100.fits'%(ulirg.value, ulirg.value,filter_name, ))#, data = d1, overwrite= True)
    err = fits.getdata(data_dir+'ULIRG%s/gal%s_UV_%s_err_x550_y550_sz100.fits'%( ulirg.value,ulirg.value,filter_name,))#, data = d3, overwrite= True)
    A = fits.getdata(data_dir+'ULIRG%s/A_lmfit_x550_y550_sz100.fits'%(ulirg.value))#, data = d3, overwrite= True)
    b = fits.getdata(data_dir+'ULIRG%s/b_lmfit_x550_y550_sz100.fits'%(ulirg.value))#, data = d3, overwrite= True)
    m = fits.getdata(data_dir+'ULIRG%s/m_lmfit_x550_y550_sz100.fits'%(ulirg.value))#, data = d3, overwrite= True)

    A_err = fits.getdata(data_dir+'ULIRG%s/A_lmfit_err_x550_y550_sz100.fits'%(ulirg.value))#, data = d3, overwrite= True)
    b_err = fits.getdata(data_dir+'ULIRG%s/b_lmfit_err_x550_y550_sz100.fits'%(ulirg.value))#, data = d3, overwrite= True)
    m_err = fits.getdata(data_dir+'ULIRG%s/m_lmfit_err_x550_y550_sz100.fits'%(ulirg.value))#, data = d3, overwrite= True)



    A_cut = A[550:650, 550:650]
    b_cut = b[550:650, 550:650]
    m_cut = m[550:650, 550:650]


    A_err_cut = A_err[550:650, 550:650]
    b_err_cut = b_err[550:650, 550:650]
    m_err_cut = m_err[550:650, 550:650]

    TOOLS = " pan,lasso_select, box_select,help,wheel_zoom,box_zoom,reset, tap, undo, redo"
    #taptool = TapTool(callback=callback)

    print (A_cut.shape, err.shape)


    p = figure(title = 'ULIRG%s, %s '%(ulirg.value,filter_name),tools=TOOLS, plot_width=300, plot_height=300,x_range=x_range, y_range= y_range,
               tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image"), ("err", "@err"),("LYA", "@A_cut")  ])

    # must give a vector of image data for image parameter

    data_all = dict(image= [data], err =[err], A_cut = [A_cut] )
    
    color_mapper = LogColorMapper(palette="Viridis256", low=1e-3, high=1e-2*0.5)

    p.image(source=data_all,image ='image', x=0, y=0, dw=100, dh=100, color_mapper= color_mapper)
    

    p.x([float(x_plot.value)], [float(y_plot.value)], size=25, color="red",alpha =2.0)







    return p, data,err




def table_values(ulirg, x_plot, y_plot):
    A = fits.getdata(data_dir+'ULIRG%s/A_lmfit_x550_y550_sz100.fits'%(ulirg.value))#, data = d3, overwrite= True)
    b = fits.getdata(data_dir+'ULIRG%s/b_lmfit_x550_y550_sz100.fits'%(ulirg.value))#, data = d3, overwrite= True)
    m = fits.getdata(data_dir+'ULIRG%s/m_lmfit_x550_y550_sz100.fits'%(ulirg.value))#, data = d3, overwrite= True)

    A_err = fits.getdata(data_dir+'ULIRG%s/A_lmfit_err_x550_y550_sz100.fits'%(ulirg.value))#, data = d3, overwrite= True)
    b_err = fits.getdata(data_dir+'ULIRG%s/b_lmfit_err_x550_y550_sz100.fits'%(ulirg.value))#, data = d3, overwrite= True)
    m_err = fits.getdata(data_dir+'ULIRG%s/m_lmfit_err_x550_y550_sz100.fits'%(ulirg.value))#, data = d3, overwrite= True)



    A_cut = A[550:650, 550:650]
    b_cut = b[550:650, 550:650]
    m_cut = m[550:650, 550:650]


    A_err_cut = A_err[550:650, 550:650]
    b_err_cut = b_err[550:650, 550:650]
    m_err_cut = m_err[550:650, 550:650]

    x1 = int(x_plot.value)
    y1 = int(y_plot.value)
    dict_all = OrderedDict({'params':params, 'values':[ulirg.value, x1,y1,format_all(A_cut[x1,y1]),\
        format_all(b_cut[x1,y1]),format_all(m_cut[x1,y1]), format_all(A_err_cut[x1,y1]), format_all(b_err_cut[x1,y1]),format_all(m_err_cut[x1,y1])]})
    source = ColumnDataSource(dict_all)



    columns = [
            TableColumn(field="params", title="params"),
            TableColumn(field="values", title="values"),
        ]
    data_table = DataTable(source=source, columns=columns)



    return data_table


def create_figure():

    p1, data1, err1 = plot_lya(data_dir,'F125',x_range=(0, 100), y_range=(0, 100))
    p2, data2, err2 = plot_lya(data_dir,'F140',x_range= p1.x_range,y_range=p1.y_range )
    p3, data3, err3 = plot_lya(data_dir,'F150',x_range= p1.x_range,y_range=p1.y_range )
    p4, data4, err4 = plot_lya(data_dir,'F165',x_range= p1.x_range,y_range=p1.y_range )


    TOOLS = " pan,lasso_select, box_select,help,wheel_zoom,box_zoom,reset, tap, undo, redo"


    color_mapper = LogColorMapper(palette="Viridis256", low=1e-3, high=1e-2*0.5)
    color_bar = ColorBar(color_mapper=color_mapper, ticker=LogTicker(), width =5,major_label_text_font_size="5pt",
                         label_standoff=12, border_line_color=None, location=(0,0))

    p4.add_layout(color_bar, 'center')



    x1 = int(x_plot.value)
    y1 = int(y_plot.value)

    p5 = figure(tools = TOOLS,title='errorbars with UV filters and fits', width=300, height=300)

    p5.xaxis.axis_label = 'wavelength'
    p5.yaxis.axis_label = 'counts'


    xs = np.array([1438.19187645, 1527.99574111, 1612.22929754, 1762.54619064])

    ys = [data1[x1,y1],data2[x1,y1],data3[x1,y1], data4[x1,y1]]
    yerrs = [err1[x1,y1],err2[x1,y1],err3[x1,y1], err4[x1,y1]]

    p5.circle(xs, ys, color='green', size=10, line_alpha=0)


    err_xs = []
    err_ys = []

    for x, y, yerr in zip(xs, ys, yerrs):
        err_xs.append((x, x))
        err_ys.append((y - yerr, y + yerr))

    p5.multi_line(err_xs, err_ys, color='black')

    # show(p)



    A = fits.getdata(data_dir+'ULIRG%s/A_lmfit_x550_y550_sz100.fits'%(ulirg.value))#, data = d3, overwrite= True)
    b = fits.getdata(data_dir+'ULIRG%s/b_lmfit_x550_y550_sz100.fits'%(ulirg.value))#, data = d3, overwrite= True)
    m = fits.getdata(data_dir+'ULIRG%s/m_lmfit_x550_y550_sz100.fits'%(ulirg.value))#, data = d3, overwrite= True)


    A_err = fits.getdata(data_dir+'ULIRG%s/A_lmfit_err_x550_y550_sz100.fits'%(ulirg.value))#, data = d3, overwrite= True)

    A_cut = A[550:650, 550:650]
    b_cut = b[550:650, 550:650]
    m_cut = m[550:650, 550:650]

    A_err_cut = A[550:650, 550:650]


    y_fitted, spec = counts_out(m_cut[x1,y1], b_cut[x1,y1], A_cut[x1, y1])

    p5.x(xs, y_fitted, color='red', size=15)


    p6 = figure(tools= TOOLS, title='spectrum',x_range=(1200, 1900),\
     width=300, height=300, y_axis_type="log",x_axis_type="log", background_fill_color="#fafafa")

    p6.xaxis.axis_label = 'wavelength'
    p6.yaxis.axis_label = 'flux'
    
    wave1 = np.arange(1000, 10000, 1.0)

    p6.line(wave1, spec, color = 'blue',line_width =1.0)





    p7 = figure(title = 'ULIRG%s, LYA '%(ulirg.value),tools=TOOLS, plot_width=300, plot_height=300,x_range=p1.x_range, y_range= p1.y_range,
                   tooltips=[("x", "$x"), ("y", "$y"), ("LYA", "@image") , ("ERR", "@A_err_cut")  ])

    # must give a vector of image data for image parameter

    data_lya = dict(image= [A_cut], A_err_cut =[A_err_cut] )
    
    color_mapper = LogColorMapper(palette="Greys256")#, low=1e-3, high=1e-2*0.5)

    p7.image(source=data_lya,image ='image', x=0, y=0, dw=100, dh=100, color_mapper= color_mapper)
    
    p7.x([float(x_plot.value)], [float(y_plot.value)], size=25, color="red",alpha =2.0)



    p = gridplot([[p1, p2,p6], [p3, p4, p5], [p7]])



    return row(p, column(widgetbox(button), widgetbox(table_values(ulirg, x_plot, y_plot)), widgetbox(pre)))

def update(attr, old, new):

    layout.children[1] = row(create_figure())





ulirg = Select(title='Select ULIRG:', value="1", options= ulirg_num)
ulirg.on_change('value', update)

x_plot = TextInput(value="60", title="x_value:")
x_plot.on_change('value', update)

y_plot = TextInput(value="75", title="y_value:")
y_plot.on_change('value', update)


controls =column(widgetbox([ulirg, x_plot, y_plot]), width=250)
layout = row(controls, create_figure())

curdoc().add_root(layout)
curdoc().title = "LYA ULIRG plotting "