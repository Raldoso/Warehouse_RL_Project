from enviroment import WarehouseEnv, Store
import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm
from agent import Agent
import numpy as np


N_DAYS = 300
""" 
SETUP ENVIROMENT
"""
env = WarehouseEnv(
    max_age=6,
    n_days=N_DAYS
    )
env.addStore(
    Store(
        avg_range=[8],
        std_range=[5],
        max_age=6,
        min_items_percent=0.1))
env.addStore(
    Store(
        avg_range=[13],
        std_range=[5],
        max_age=6,
        min_items_percent=0.1))
env.addStore(
    Store(
        avg_range=[20],
        std_range=[5],
        max_age=6,
        min_items_percent=0.1))
env.setup_spaces()

""" 
RELOAD MODEL AND TEST AGENT
"""
agent = Agent(
    state_size=env.state_size,
    action_size=env.maxorder)
agent.load_model(r"C:\Users\ASUS\Downloads\ItWork\Projects\Udemy_PyML_Bootc\LIDL_ML_Procect\models\360_(105493.0)_warehouse_agent.pth")

scores = []
q_history = np.zeros((N_DAYS,env.maxorder))
#interesting model (360_(105493.0)_warehouse_agent.pth)


state = env.reset()
for day in tqdm(range(N_DAYS)):
    action, q_values = agent.choose_action(state,simulate=True)
    q_history[day] = q_values-np.mean(q_values)+1000
    state, reward, done, error = env.step(action)
    if done:break



""" 3D Surface Plot """

def plot_Q_surface(q_history):
    fig = go.Figure(data=[go.Surface(z=q_history)])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                    highlightcolor="limegreen", project_z=True))
    fig.update_layout(title='Q history', autosize=False,
                    scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),
                    width=650, height=650,
                    margin=dict(l=65, r=50, b=65, t=90))
    fig.update_scenes(xaxis_title_text='Order',  
                    yaxis_title_text='Day',  
                    zaxis_title_text='Q value')
    fig.show()

def plot_Q_contour(q_history):
    fig = go.Figure(data =
        go.Contour(
            z=q_history.T,
            colorbar=dict(nticks=20, 
                          ticks='outside',
                          ticklen=5, 
                          tickwidth=1,
                          showticklabels=True,
                          tickangle=0, 
                          tickfont_size=12)
        ))
    fig.update_xaxes(range=(0,q_history.shape[0]))
    fig.update_yaxes(range=(0,q_history.shape[1]))
    fig.update_layout(
        title="Q history",
        xaxis_title_text='Day',
        yaxis_title_text='Order')
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=np.arange(N_DAYS),
            y=np.argmax(q_history,axis=1),
            marker=dict(
                color='rgb(0, 255, 0)',
                size=3,
                line=dict(
                    color='rgb(0, 102, 204)',
                    width=1
                    )
            ),
            showlegend=False))
    fig.show()


fig = make_subplots(rows=4,
                    cols=2,
                    vertical_spacing=0.09,
                    horizontal_spacing=0.05,
                    row_heights=[0.2, 0.2, 0.2, 0.4],
                    subplot_titles=["3D Q-value plot","Recieved","Storage","Error","Up view of Q-value plot"],
                    specs=[[{"type": "surface","rowspan":3},{"type": "scatter"}],
                                    [None,                  {"type": "scatter"}],
                                    [None,                  {"type": "scatter"}],
                                    [{"type": "contour"},   {"type": "scatter"}]])

""" 3D PLOT """
fig.add_trace(
    go.Contour(
        z=q_history.T,
        ncontours=15,
        colorscale="Plasma",
        colorbar_x=-0.15
    ),
    row=4,
    col=1
)
fig.add_trace(
    go.Scatter(
        mode='markers',
        x=np.arange(N_DAYS),
        y=np.argmax(q_history,axis=1),
        marker=dict(
            color='rgb(0, 255, 0)',
            size=3,
            line=dict(
                color='rgb(0, 102, 204)',
                width=1
                )
        ),
        showlegend=False
    ),
    row=4,
    col=1
)

x=[]
y=[]
z=[]
for day in range(q_history.shape[0]):
    x.append(day)
    y.append(np.argmax(q_history[day]))
    z.append(np.max(q_history[day]))

fig.add_trace(
    go.Surface(
        z=q_history.T,
        showscale=False,
    ),
    row=1, 
    col=1
)
# fig.add_trace(
#     go.Scatter3d(
#         x=x,
#         y=y,
#         z=z,
#         mode="markers",
#         showlegend=False,
#         marker=dict(
#                 color='rgb(0, 255, 0)',
#                 size=3,
#                 line=dict(
#                     color='rgb(0, 102, 204)',
#                     width=1
#                     )
#             )
#     )
# )
# """ RECIEVED """
fig.add_trace(
    go.Scatter(
        y=env.stores[0].history[:,0],#env.stores[1].history[:,0]+env.stores[2].history[:,0],
        showlegend=False
    ),
    row=1,
    col=2
)
""" STORAGE """
fig.add_trace(
    go.Scatter(
        y=env.stores[0].history[:,1],
        showlegend=False
    ),
    row=2,
    col=2
)
""" ERROR """
fig.add_trace(
    go.Scatter(
        y=env.stores[0].history[:,5]-env.stores[0].history[:,3],
        showlegend=False
    ),
    row=3,
    col=2
)
# fig.update_xaxes(autorange="reversed",row=1,col=1)
# fig.update_layout(
#     xaxis_title='Day',
#     yaxis_title='Order',
#     zaxis_title_text='Q value',
# )
# fig.update_layout(
#     xaxis_title_text='Day',
#     yaxis_title_text='Order',
# )
fig.update_layout(
    width=920,
    height=450,
    margin=dict(l=0,t=30,r=0,b=0,pad=0),
)
fig.update_scenes(
    aspectratio=dict(x=2, y=1, z=1),
    aspectmode="manual")
fig.show()



def plot_store_simulation():
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1,constrained_layout=True)

    recieved = env.stores[0].history[:,0]
    storage = env.stores[0].history[:,1]
    bought = env.stores[0].history[:,2]
    overbuy = env.stores[0].history[:,3]
    ordered = env.stores[0].history[:,4]
    expired = env.stores[0].history[:,5]

    # plt.hist(actions, bins=env.maxorder)

    ax1.plot(recieved[4:],"tab:green")
    ax1.set_title("Recieved")
    ax1.grid()

    ax2.plot(storage,"tab:blue")
    ax2.set_title("Sum of storage")
    ax2.grid()

    ax3.plot(bought,"tab:grey")
    ax3.set_title("Daily bought amount")
    ax3.grid()

    ax4.plot(overbuy,"tab:orange")
    ax4.set_title("Overbuy")
    ax4.grid()

    ax5.plot(ordered,"tab:purple")
    ax5.set_title("Store status")
    ax5.grid()

    ax6.plot(expired,"tab:red")
    ax6.set_title(" Daily expired items")
    ax6.grid()

    plt.show()