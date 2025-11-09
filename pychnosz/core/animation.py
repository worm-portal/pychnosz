import plotly.express as px
import plotly.graph_objects as go
import copy
import numpy as np
import pandas as pd
import statistics

# Import chnosz functions
from .basis import basis
from .species import species
from .affinity import affinity
from .equilibrate import equilibrate
from .diagram import diagram, diagram_interactive
from .info import info

# Import chemlabel for formatting chemical formulas
try:
    from WORMutils import chemlabel
except ImportError:
    # Fallback: simple identity function if WORMutils is not available
    def chemlabel(s):
        """Simple fallback for chemical label formatting."""
        return str(s)


def __seq(start, end, by=None, length_out=None):
    
    """
    Mimic the seq() function in base R.
    """
    
    len_provided = True if (length_out is not None) else False
    by_provided = True if (by is not None) else False
    if (not by_provided) & (not len_provided):
        raise ValueError('At least by or n_points must be provided')
    width = end - start
    eps = pow(10.0, -14)
    if by_provided:
        if (abs(by) < eps):
            raise ValueError('by must be non-zero.')
        absby = abs(by)
        if absby - width < eps: 
            length_out = int(width / absby)
        else: 
            # by is too great, we assume by is actually length_out
            length_out = int(by)
            by = width / (by - 1)
    else:
        length_out = int(length_out)
        by = width / (length_out - 1) 
    out = [float(start)]*length_out
    for i in range(1, length_out):
        out[i] += by * i
    if abs(start + by * length_out - end) < eps:
        out.append(end)
    return out


def animation(basis_args={}, species_args={}, affinity_args={},
              equilibrate_args=None, diagram_args={},
              anim_var="T", anim_range=[0, 350, 8], xlab=None, ylab=None,
              save_as="newanimationframe", save_format="png", height=300,
              width=400, save_scale=1,
              messages=False):
    
    """
    Produce an animated interactive affinity, activity, or predominance diagram.
    
    Parameters
    ----------
    basis_args : dict
        Dictionary of options for defining basis species (see `basis`) in the
        animated diagram.
        Example: basis_args={'species':['CO2', 'O2', 'H2O', 'H+']}

    species_args : dict
        Dictionary of options for defining species (see `species`) in the
        animated diagram, or a list of dicts.
        Example 1: species_args={'species':['CO2', 'HCO3-', 'CO3-2']}
        Example 2: species_args=[
                {'species':['CO2', 'HCO3-', 'CO3-2'], 'state':[-4]},
                {'species':['graphite'], state:[0], 'add':True}]

    affinity_args : dict
        Dictionary of options for defining the affinity calculation (see
        `affinity`).
        Example: affinity_args={"pH":[2, 12, 100]}
        Example: affinity_args={"pH":[2, 12, 100], "P":[2000, 4000, 100]}
    
    equilibrate_args : dict or None, default None
        Dictionary of options for defining equilibration calculation
        (see `equilibrate`). If None, plots output from `affinity`.
        Example: equilibrate_args={"balance":1}
    
    diagram_args : dict
        Dictionary of options for diagramming (see `diagram`). Diagram option
        `interactive` is set to True.
        Example: diagram_args={"alpha":True}
    
    anim_var : str, default "T"
        Variable that changes with each frame of animation.
    
    anim_range : list of numeric, default [0, 350, 8]
        The first two numbers in the list are the starting and ending
        values for `anim_var`. The third number in the list is the desired
        number of animation frames.

    xlab, ylab : str, optional
        Custom names for the X and Y axes.
    
    messages : bool, default True
        Display messages from CHNOSZ?
    
    Returns
    -------
    An interactive animated plot.
    """
    
    # cap number of frames in animation. Remove limitation after more testing.
    if isinstance(anim_range, list):
        if len(anim_range) == 3:
            if anim_range[2] > 30:
                raise Exception("anim_range is limited to 30 frames.")
        else:
            raise Exception("anim_range must be a list with three values: starting "
                            "value of anim_var, stopping value, and number of "
                            "frames in the animation")
    else:
        raise Exception("anim_range must be a list with three values: starting "
                        "value of anim_var, stopping value, and number of "
                        "frames in the animation")

    if isinstance(basis_args, dict):
        if "species" not in basis_args.keys():
            raise Exception("basis_args needs to contain a list of species for 'species'. "
                            "Example: basis_args={'species':['CO2', 'O2', 'H2O', 'H+']}")
    else:
        raise Exception("basis_args needs to be a Python dictionary with a key "
                        "called 'species' (additional keys are optional). "
                        "Example: basis_args={'species':['CO2', 'O2', 'H2O', 'H+']}")


    # Add messages parameter to basis_args if not already present
    if "messages" not in basis_args.keys():
        basis_args["messages"] = messages

    basis_out = basis(**basis_args)
    basis_sp = list(basis_out.index)
    basis_state = list(basis_out["state"])
    
    if isinstance(species_args, dict):
        if "species" not in species_args.keys():
            raise Exception("species_args needs to contain a list of species for 'species'. "
                            "Example: species_args={'species':['CO2', 'HCO3-', 'CO3-2']}")
        species_args_list = [species_args]
    elif isinstance(species_args, list):
        species_args_list = species_args
        for species_args in species_args_list:
            if "species" not in species_args.keys():
                raise Exception("species_args needs to contain a list of species for 'species'. "
                                "Example: species_args={'species':['CO2', 'HCO3-', 'CO3-2']}")
    else:
        raise Exception("species_args needs to be either a Python dictionary with a key "
                        "called 'species' (additional keys are optional). "
                        "Example: species_args={'species':['CO2', 'HCO3-', 'CO3-2']}"
                        "or else species_args needs to be a list of Python dictionaries."
                        "Example: species_args=[{'species':['CO2', 'HCO3-', 'CO3-2'], 'state':[-4]},"
                        "{'species':['graphite'], state:[0], 'add':True}]")

    # There may be multiple arguments passed to species, especially in cases
    # where add=True. Loop through all the arguments to apply them.
    for species_args in species_args_list:
        if "logact" in species_args.keys():
            mod_species_logact = copy.copy(species_args['logact'])
            del species_args['logact']
        else:
            mod_species_logact = []

        # Add messages parameter to species_args if not already present
        if "messages" not in species_args.keys():
            species_args["messages"] = messages

        species_out = species(**species_args)

        if len(mod_species_logact)>0:
            for i in range(0, len(mod_species_logact)) :
                species_out = species(species_args["species"][i], mod_species_logact[i], messages=messages)

    sp = list(species_out["name"])

    if isinstance(sp[0], (int, np.integer)):
        sp = [info(s, messages=messages)["name"].values[0] for s in sp]

    dfs = []
    dmaps = []
    dmaps_names = []
    
    if len(anim_range) == 2:
        anim_res = 8
        anim_range = anim_range + [anim_res]
    elif len(anim_range) == 3:
        anim_res = anim_range[2]
        anim_range = [anim_range[0], anim_range[1]]
    
    zvals = __seq(anim_range[0], anim_range[1], length_out=anim_res)
        
    if "messages" not in affinity_args.keys():
        affinity_args["messages"] = messages
    if "messages" not in diagram_args.keys():
        diagram_args["messages"] = messages
    if "plot_it" not in diagram_args.keys():
        diagram_args["plot_it"] = False
    diagram_args["interactive"] = True
    if "format_names" not in diagram_args.keys():
        format_names=True
        format_x_names=True
        format_y_names=True
    
    for z in zvals:

        if anim_var in basis_out.index:
            basis_out = basis(anim_var, z, messages=messages)
        elif anim_var in list(species_out["name"]):
            species_out = species(anim_var, -z, messages=messages)
        elif anim_var == "pH":
            basis_out = basis("H+", -z, messages=messages)
        else:
            affinity_args[anim_var] = z
        
        aeout = affinity(**affinity_args)

        if equilibrate_args != None:
            equilibrate_args["aout"] = aeout
            if "messages" not in equilibrate_args.keys():
                equilibrate_args["messages"] = messages
            aeout = equilibrate(**equilibrate_args)

        # Get affinity arguments from the result dictionary
        aeout_args = aeout.get("args", {})
        xvar = list(aeout_args.keys())[0]
        xrange = list(aeout_args[xvar])

        res_default = 256 # default affinity resolution
        if len(xrange) == 3:
            xres = int(xrange[2])
        else:
            xres = res_default
        
        diagram_args["eout"] = aeout

        # Use diagram_interactive since interactive=True is set
        # Remove 'interactive' key as diagram_interactive doesn't need it
        diagram_args_copy = diagram_args.copy()
        diagram_args_copy.pop('interactive', None)
        df, fig = diagram_interactive(**diagram_args_copy)

        # Check if this is a predominance plot (2D) or affinity/activity plot (1D)
        if 'pred' not in df.columns:
            # affinity/activity plot (1D) - melt to long format for animation
            is_predom_plot = False
            id_vars = [xvar]  # Keep the x-variable as identifier
            value_vars = [col for col in df.columns if col != xvar]  # All species columns
            df_melted = df.melt(id_vars=id_vars, value_vars=value_vars,
                               var_name='variable', value_name='value')
            df_melted[anim_var] = z
            dfs.append(df_melted)
        else:
            # predominance plot (2D) - keep original format with pred and prednames
            is_predom_plot = True
            df[anim_var] = z
            dfs.append(df)
            yvar = list(aeout_args.keys())[1]
            yrange = list(aeout_args[yvar])
            if len(yrange) == 3:
                yres = int(yrange[2])
            else:
                yres = res_default

            data = np.array(df.pred)
            shape = (xres, yres)
            dmap = data.reshape(shape)
            dmaps.append(dmap)

            data = np.array(df.prednames)
            shape = (xres, yres)
            dmap_names = data.reshape(shape)
            dmaps_names.append(dmap_names)
        
    xvals = __seq(xrange[0], xrange[1], length_out=xres)


    unit_dict = {"P":"bar", "T":"°C", "pH":"", "Eh":"volts", "IS":"mol/kg"}
    
    if any([anim_var in basis_out.index, anim_var in list(species_out["name"])]) and anim_var not in unit_dict.keys():
        unit_dict[anim_var] = "logact "+anim_var

    for i,s in enumerate(basis_sp):
        if basis_state[i] in ["aq", "liq", "cr"]:
            if format_names:
                unit_dict[s] = "log <i>a</i><sub>{}</sub>".format(chemlabel(s))
            else:
                unit_dict[s] = "log <i>a</i><sub>{}</sub>".format(s)
        else:
            if format_names:
                unit_dict[s] = "log <i>f</i><sub>{}</sub>".format(chemlabel(s))
            else:
                unit_dict[s] = "log <i>f</i><sub>{}</sub>".format(s)

    xlab = xvar+", "+unit_dict[xvar]
    
    if xvar in basis_sp:
        xlab = unit_dict[xvar]
    if xvar == "pH":
        xlab = "pH"
    
    if is_predom_plot:
        ylab = yvar+", "+unit_dict[yvar]
        if yvar in basis_sp:
            ylab = unit_dict[yvar]
        if yvar == "pH":
            yvar = "pH"

    
    if not is_predom_plot:

        if 'loga.equil' not in aeout.keys():
            yvar = "A/(2.303RT)"
        else:
            yvar = "log a"
        if "alpha" in diagram_args.keys():
            if diagram_args["alpha"]:
                yvar = "alpha"
        
        df_c = pd.concat(dfs)

        if "fill" in diagram_args.keys():
            if isinstance(diagram_args["fill"], list):
                colormap = {key:col for key,col in zip(list(dict.fromkeys(df_c["variable"])), diagram_args["fill"])}
            else:
                colormap = diagram_args["fill"]

            # with color mapping
            fig = px.line(df_c, x=xvar, y="value", color='variable', template="simple_white",
                          width=500,  height=400, animation_frame=anim_var,
                          color_discrete_map = colormap,
                          labels=dict(value=yvar, x=xvar),
                         )
        else:
            # without color mapping
            fig = px.line(df_c, x=xvar, y="value", color='variable', template="simple_white",
                          width=500,  height=400, animation_frame=anim_var,
                          labels=dict(value=yvar, x=xvar),
                         )
        
        if "annotation" in diagram_args.keys():
            if "annotation_coords" not in diagram_args.keys():
                diagram_args["annotation_coords"] = [0, 0]
            fig.add_annotation(x=diagram_args["annotation_coords"][0],
                               y=diagram_args["annotation_coords"][1],
                               xref="paper",
                               yref="paper",
                               align='left',
                               text=diagram_args["annotation"],
                               bgcolor="rgba(255, 255, 255, 0.5)",
                               showarrow=False)
        
        if 'main' in diagram_args.keys():
            fig.update_layout(title={'text':diagram_args["main"], 'x':0.5, 'xanchor':'center'})

        if isinstance(xlab, str):
            fig.update_layout(xaxis_title=xlab)
        if isinstance(ylab, str):
            fig.update_layout(yaxis_title=ylab)
        
        if 'fill' in diagram_args.keys():
            if isinstance(diagram_args["fill"], list):
                for i,v in enumerate(diagram_args["fill"]):
                    fig['data'][i]['line']['color']=v
        
        fig.update_layout(legend_title=None)

        config = {'displaylogo': False,
                  'modeBarButtonsToRemove': ['resetScale2d', 'toggleSpikelines'],
                  'toImageButtonOptions': {
                                             'format': save_format, # one of png, svg, jpeg, webp
                                             'filename': save_as,
                                             'height': height,
                                             'width': width,
                                             'scale': save_scale,
                                          },
                 }

        fig.show(config=config)
        return
    
    else:
        yvals = __seq(yrange[0], yrange[1], length_out=yres)


    
    frames = []
    slider_steps = []
    annotations = []
    cst_data = []
    heatmaps = []
    
    # i is a frame in the animation
    for i in range(0, len(zvals)):

        annotations_i = []
        for s in sp:
            if s in set(dfs[i]["prednames"]):
                # if an annotation should appear, create one for this frame
                df_s = dfs[i].loc[dfs[i]["prednames"]==s,]
                namex = df_s[xvar].mean()
                namey = df_s[yvar].mean()
                a = go.layout.Annotation(
                    x=namex,
                    y=namey,
                    xref="x",
                    yref="y",
                    text=chemlabel(s),
                    bgcolor="rgba(255, 255, 255, 0.5)",
                    showarrow=False,
                    )
            else:
                # if an annotation shouldn't appear, make an invisible annotation
                # (workaround for a plotly bug where annotations won't clear in an animation)
                namex = statistics.mean(xvals)
                namey = statistics.mean(yvals)
                a = go.layout.Annotation(
                    x=namex,
                    y=namey,
                    xref="x",
                    yref="y",
                    text="",
                    bgcolor="rgba(255, 255, 255, 0)",
                    showarrow=False,
                    )
            annotations_i.append(a)
            
        # allows adding a custom annotation; append to frame
        if "annotation" in diagram_args.keys():
            if "annotation_coords" not in diagram_args.keys():
                diagram_args["annotation_coords"] = [0, 0]
            custom_annotation = go.layout.Annotation(
                    x=diagram_args["annotation_coords"][0],
                    y=diagram_args["annotation_coords"][1],
                    xref="paper",
                    yref="paper",
                    align='left',
                    text=diagram_args["annotation"],
                    bgcolor="rgba(255, 255, 255, 0.5)",
                    showarrow=False,
                    )
            annotations_i.append(custom_annotation)
    
        annotations.append(annotations_i)

        if 'ylab' in diagram_args.keys():
            ylab = diagram_args["ylab"]
            hover_ylab = ylab+': %{y} '
        else:
            ylab = chemlabel(ylab)
            hover_ylab = yvar+': %{y} '+unit_dict[yvar]

        if 'xlab' in diagram_args.keys():
            xlab = diagram_args["xlab"]
            hover_xlab = xlab+': %{x} '
        else:
            xlab = chemlabel(xlab)
            hover_xlab = xvar+': %{x} '+unit_dict[xvar]
        
        heatmaps_i = go.Heatmap(z=dmaps[i], x=xvals, y=yvals, zmin=0, zmax=len(sp)-1,
                                customdata=dmaps_names[i],
                                hovertemplate=hover_xlab+'<br>'+hover_ylab+'<br>Region: %{customdata}<extra></extra>')

        heatmaps.append(heatmaps_i)

        frame = go.Frame(data=[heatmaps_i],
                         name=str(i),
                         layout=go.Layout(annotations=annotations_i))

        frames.append(frame)

        slider_step = dict(
            method='animate',
            label=zvals[i],
            value=i,
            args=[
                [i],
                dict(
                    frame=dict(duration=300, redraw=True),
                    mode='immediate',
                    transition=dict(duration=0)
                )
            ]
        )

        slider_steps.append(slider_step)

    fig = go.Figure(
        data = heatmaps[0],
        layout=go.Layout(
    #         title="Frame 0",
            title_x=0.5,
            width=500, height=500,
            annotations=annotations[0],
            sliders=[dict(
                active=0,
                yanchor='top',
                xanchor='left',
                currentvalue=dict(
                    font=dict(size=12),
                    prefix='{}: '.format(anim_var),
                    suffix=' '+unit_dict[anim_var],
                    visible=True,
                    xanchor='right'
                ),
                transition=dict(duration=0, easing='cubic-in-out'),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.1,
                y=0,
                steps=slider_steps
            )],
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                              method="animate",
                              args=[None, {"fromcurrent":True}]),
                        dict(label="Pause",
                             method="animate",
                             args=[[None],
                                   {"frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                             )],
                direction="left",
                pad={"r": 10, "t": 87},
                showactive=False,
                x=0.1,
                xanchor="right",
                y=0,
                yanchor="top",
            )]
        ),
        frames=frames

    )


    if 'fill' in diagram_args.keys():
        if isinstance(diagram_args["fill"], list):
            colorscale_temp = []
            for i,v in enumerate(diagram_args["fill"]):
                colorscale_temp.append([i, v])
            colorscale = colorscale_temp
        elif isinstance(diagram_args["fill"], str):
            colorscale = diagram_args["fill"]
    else:
        colorscale = "viridis"
    
    fig.update_traces(dict(showscale=False,
                           colorscale=colorscale),
                      selector={'type':'heatmap'})
    
    fig.update_layout(
        xaxis_title=xlab,
        yaxis_title=ylab,
        xaxis={"range":[list(dfs[0][xvar])[0], list(dfs[0][xvar])[-1]]},
        yaxis={"range":[list(dfs[0][yvar])[0], list(dfs[0][yvar])[-1]]},
        margin={"t": 60, "r":60},
    )

    if 'main' in diagram_args.keys():
        fig.update_layout(title={'text':diagram_args['main'], 'x':0.5, 'xanchor':'center'})

    config = {'displaylogo': False,
              'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d',
                                         'autoScale2d', 'toggleSpikelines',
                                         'hoverClosestCartesian', 'hoverCompareCartesian'],
              'toImageButtonOptions': {
                                       'format': save_format, # one of png, svg, jpeg, webp
                                       'filename': save_as,
                                       'height': height,
                                       'width': width,
                                       'scale': save_scale,
                                        },
             }
    

    fig.show(config=config)