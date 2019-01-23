import plotly
import plotly.plotly as py
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
plotly.tools.set_credentials_file(username='rohanpaleja27', api_key='kjaKFX2DQpnvmcHR8QUU')

class Timeline:
    def __init__(self):
        self.timeline_dict = {}
        self.easy_timeline_dict = {}
        self.list_elements = []

    def add_to_timeline(self, start_time,finish_time,agent, nearest_task):
        agent_name = agent.getName()
        nearest_task_name = nearest_task.getName()
        self.timeline_dict[start_time] = [(agent_name,nearest_task_name, 'start')]
        self.timeline_dict[finish_time] = [(agent_name, nearest_task_name, 'finish')]
        self.easy_timeline_dict[start_time] = [(agent_name, nearest_task_name, finish_time)]
        self.list_elements.append([agent_name, nearest_task_name, start_time, finish_time])

    def sort_timeline(self):
        self.timeline_dict = sorted(self.timeline_dict.items())
        self.easy_timeline_dict = sorted(self.easy_timeline_dict.items())
        return self.timeline_dict

    def create_gant_chart(self):
        df = []
        for element in self.list_elements:
            df.append(dict(Task = element[1], Start = element[2], Finish = element[3], Resource = element[0]))
        print(df)
        colors = ['rgb(10,10,10)', 'rgb(150,150,150)']
        fig = ff.create_gantt(df, colors = colors, index_col = 'Resource', reverse_colors=True, show_colorbar=True)
        py.plot(fig,filename = 'gantt-string-variable', world_readable= True)

    def plot_path(self,agents):
        colors = ['black', 'blue', 'green', 'red']
        labels = []
        for agent in agents:
            agent_name = agent.getName()
            labels.append(agent_name)
        color_counter = 0
        for agent in agents:
            verts = []
            tl = agent.task_list
            verts.append(agent.orig_location)
            for task in tl:
                pos = task.getloc()
                verts.append(pos)
            xs,ys = zip(*verts)
            plt.plot(xs,ys, 'x--', lw = 2, color=colors[color_counter],ms=10, label = labels[color_counter])
            color_counter+=1
        plt.legend()
        plt.show()