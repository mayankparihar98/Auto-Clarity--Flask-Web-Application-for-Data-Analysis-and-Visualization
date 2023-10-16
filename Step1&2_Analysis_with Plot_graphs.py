# Importing all neccessary Libraries for the Project Name: Auto Clarity
import io
import os
import re
import logging
import base64
from flask import (
                    Flask,
                    make_response,
                    render_template,
                    redirect,
                    send_from_directory,
                    url_for,
                    send_file,
                    jsonify,
)
from flask_wtf import FlaskForm
from flask import session
from wtforms import FileField, SubmitField, SelectField, BooleanField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import ColumnDataSource
from bokeh.transform import linear_cmap
from bokeh.models import HelpTool, HoverTool, Title
from bokeh.palettes import Viridis256, viridis
from bokeh.layouts import gridplot

# Create a Flask application
def create_app():
    """
    Create and configure the Flask app.

    Returns:
    - app (Flask): The configured Flask app.
    """    
    app = Flask(__name__)

    # Set up logging
    setup_logging()

    # Define a custom filter function for base64 encoding
    def base64_encode(data):
        return base64.b64encode(data).decode('utf-8')

    # Add the custom filter to the Jinja environment
    app.jinja_env.filters['b64encode'] = base64_encode

    app.config['SECRET_KEY'] = 'supersecretkey'
    app.config['UPLOAD_FOLDER'] = 'Dataset'

    # Configure a route for serving static files
    @app.route('/static/<path:filename>')
    def static_file(filename):
        return send_from_directory(app.config['STATIC_FOLDER'], filename)

    @app.errorhandler(500)
    def internal_server_error(error):
        return make_response(jsonify({'error': 'Internal error occurred: Please try again'}), 500)

    # Define the logger as a global variable
    global logger
    logger = logging.getLogger(__name__)

    # Create a Flask form for uploading the file
    class UploadFileForm(FlaskForm):
        """
        A Flask form for uploading a file.
        """
        file = FileField("File", validators=[InputRequired()]) # The file input field
        submit = SubmitField("Upload File") # The submit button

    # Route for the home page and initial file upload form
    @app.route('/', methods=['GET', 'POST'])
    @app.route('/home', methods=['GET', 'POST'])
    def home():
        form = UploadFileForm()
        
        if form.validate_on_submit():
            file = form.file.data
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            session['filename'] = filename        
            return redirect(url_for('analyze', filename=filename))  # Redirect to the 'analyze' route

        return render_template('upload_form.html', form=form)

    # Function to generate the Seaborn heatmap
    def generate_missing_values_heatmap(data):
        plt.figure(figsize=(20, 5))  # Set the figure size
        sns.heatmap(data.isnull(), cmap='viridis', cbar=False, yticklabels=False)
        plt.title("Missing Values Heatmap")
        plt.xlabel("Columns")
        plt.ylabel("Rows")

        # Save the Seaborn heatmap as an image in memory (BytesIO)
        heatmap_image_buffer = io.BytesIO()
        plt.savefig(heatmap_image_buffer, format='png')
        plt.close()

        # Rewind the buffer to the beginning
        heatmap_image_buffer.seek(0)

        # Return the heatmap image data
        return heatmap_image_buffer.getvalue()

    # Create a Flask form for selecting the data for split_and_clean
    class DataSelectionForm(FlaskForm):
        data_selection = SelectField("Select Data for split_and_clean")
        clean_all_columns = BooleanField("Clean All Columns")
        submit = SubmitField("Apply split_and_clean")

    # Route for analyzing the uploaded data and displaying results
    @app.route('/analyze/<filename>', methods=['GET', 'POST'])
    def analyze(filename):
        try:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            data = pd.read_csv(file_path)
            logger.info("File read successfully")

            # Create a data selection form and populate it with column names
            data_selection_form = DataSelectionForm()
            data_selection_form.data_selection.choices = [(column, column) for column in data.columns]

            if data_selection_form.validate_on_submit():
                selected_column = data_selection_form.data_selection.data
                clean_all_columns = data_selection_form.clean_all_columns.data

                # Split and clean the data based on user choice
                data = split_and_clean(data, selected_column, clean_all_columns)

            # Steps 1-5: Data Analysis and Visualization
            column_description = data.head(30).to_string()
            data_shape = data.shape

            # Capture data.info() as a string
            data_info_buffer = io.StringIO()
            data.info(buf=data_info_buffer)
            data_info = data_info_buffer.getvalue()
            data_info_buffer.close()

            data_describe = data.describe()
            missing_values = data.isnull().sum()

            # Generate the Seaborn heatmap using the function
            heatmap_image_data = generate_missing_values_heatmap(data)

            # Pass the heatmap image data to the template
            return render_template('results.html', data=data, data_selection_form=data_selection_form,
                                column_description=column_description, data_shape=data_shape,
                                data_info=data_info, data_describe=data_describe,
                                missing_values=missing_values, filename=filename,
                                missing_values_heatmap=heatmap_image_data)

        except Exception as e:
            logger.error(f"Exception occurred: {str(e)}")
            return make_response(jsonify('Internal error occurred: Please try again'), 500)

    @app.route('/download_split_and_clean/<filename>', methods=['GET'])
    def download_split_and_clean(filename):
        try:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            data = pd.read_csv(file_path)

            # Apply split_and_clean function to the data
            data_cleaned = split_and_clean(data)

            # Create a temporary CSV file to store the split and cleaned data
            temp_file = os.path.join(app.config['UPLOAD_FOLDER'], 'split_and_cleaned_data.csv')
            data_cleaned.to_csv(temp_file, index=False)

            # Send the split and cleaned data file as a downloadable response
            return send_file(temp_file, as_attachment=True, download_name='split_and_cleaned_data.csv', mimetype='text/csv')

        except Exception as e:
            logger.error(f"Exception occurred: {str(e)}")
            return make_response(jsonify('Internal error occurred: Please try again'), 500)

    # Add this new route at the end of your Flask application
    @app.route('/reset_split_and_clean/<filename>', methods=['POST'])
    def reset_split_and_clean(filename):
        try:
            session.pop('filename', None)  # Remove the filename from the session
            session.pop('clean_all_columns', None)  # Remove the clean_all_columns flag from the session
            return redirect(url_for('analyze', filename=filename))

        except Exception as e:
            logger.error(f"Exception occurred: {str(e)}")
            return make_response(jsonify('Internal error occurred: Please try again'), 500)


    # Create a Flask form for custom graph plotting
    class CustomGraphForm(FlaskForm):
        graph_type = SelectField("Graph Type", choices=[
            ('Scatter', 'Scatter Plot'),
            ('Line', 'Line Plot'),
            ('Bar', 'Bar Chart'),
            # ('Pie', 'Pie Chart'),
            ('Histogram', 'Histogram Plot'),
            # ('Boxplot', 'Boxplot'),
            ('Interval', 'Interval Chart'),
            ('TransformMarkers', 'TransformMarkers Graph'),
            ('Combined', 'Combined Plot')
        ])
        x_column = SelectField("X Axis Data Column")
        y_column = SelectField("Y Axis Data Column")
        submit = SubmitField("Plot Graph")

    # @app.route('/custom_graphs', methods=['GET', 'POST'])
    # def custom_graphs():
    #     form = CustomGraphForm()

    #     filename = session.get('filename', None)

    #     if not filename:
    #         return "Filename not found in session."

    #     try:
    #         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    #         data = pd.read_csv(file_path)
    #         columns = data.columns.tolist()  # Get column names from the dataset

    #         form.x_column.choices = [(column, column) for column in columns]
    #         form.y_column.choices = [(column, column) for column in columns]

                    
    #         if form.validate_on_submit():
    #             graph_type = form.graph_type.data
    #             x_column = form.x_column.data
    #             y_column = form.y_column.data
    #             pass

    #             try:
    #                 # Generate a custom plot based on user inputs
    #                 plot = generate_custom_plot(data, graph_type, x_column, y_column)

    #                 # Convert the plot to HTML components
    #                 script, div = components(plot)

    #                 return render_template('custom_graphs_result.html', script=script, div=div, form=form)

    #             except Exception as e:
    #                 logger.error(f"Exception occurred: {str(e)}")
    #                 return f"An exception occurred: {str(e)}"

    #     except Exception as e:
    #         logger.error(f"Exception occurred: {str(e)}")
    #         return f"An exception occurred: {str(e)}"

    #     return render_template('custom_graphs_form.html', form=form)

    # Add this list at the beginning of the script
    generated_plots = []

    # Route for the custom graphs result page
    @app.route('/custom_graphs_result', methods=['POST', 'GET'])
    def custom_graphs_result():
        try:
            filename = session.get('filename')
            if not filename:
                return "Filename not found in session."

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            data = pd.read_csv(file_path)

            form = CustomGraphForm()
            form.x_column.choices = [(column, column) for column in data.columns]
            form.y_column.choices = [(column, column) for column in data.columns]

            if form.validate_on_submit():
                graph_type = form.graph_type.data
                x_column = form.x_column.data
                y_column = form.y_column.data

                # Generate the selected plot
                plot = generate_custom_plot(data, graph_type, x_column, y_column)
                plot_script, plot_div = components(plot)

                if not generated_plots:  # If the list is empty, this is the first plot
                    generated_plots.append((x_column, y_column, plot_div, plot_script))
                else:  # For additional plots, add to the list
                    generated_plots.append((x_column, y_column, plot_div, plot_script))

                return render_template('custom_graphs_result.html', generated_plots=generated_plots, form=form)

        except Exception as e:
            logger.error(f"Exception occurred: {str(e)}")
            return make_response(jsonify('Internal error occurred: Please try again'), 500)

        return render_template('custom_graphs_result.html', generated_plots=generated_plots, form=form)

    # Add this new route at the end of the script
    @app.route('/reset_generated_plots', methods=['POST'])
    def reset_generated_plots():
        generated_plots.clear()  # Clear the list of generated plots
        return redirect(url_for('custom_graphs_result'))


    def generate_custom_plot(data, graph_type, x_column, y_column):
        plot = figure(title=f"{graph_type.capitalize()} Plot", x_axis_label=x_column, y_axis_label=y_column)
        plot.title.text = f"{graph_type.capitalize()} Plot: {x_column} vs {y_column}"    

        if graph_type == 'Scatter':
            scatter_plot(data, x_column, y_column, plot)
        
        elif graph_type == 'Bar':
            bar_chart(data, x_column, y_column, plot)

        elif graph_type == 'Line':
            line_chart(data, x_column, y_column, plot)

        elif graph_type == 'Histogram':
            histogram_plot(data, x_column, y_column, plot)

        elif graph_type == 'Interval':
            interval_chart(data, x_column, y_column, plot)

        elif graph_type == 'TransformMarkers':
            transform_markers_plot(data, x_column, y_column, plot)

        elif graph_type == 'Combined':
            generate_combined_plot(data, x_column, y_column, plot)

        customize_plot(plot)
        return plot

    def scatter_plot(data, x_column, y_column, plot):
        source = ColumnDataSource(data={x_column: data[x_column], y_column: data[y_column]})
        plot.scatter(
            x=x_column, y=y_column, source=source, size=8, color="blue", alpha=0.7,
            legend_label=f'{x_column} vs {y_column}'
        )

        hover = HoverTool()
        hover.tooltips = [(x_column, f'@{x_column}'), (y_column, f'@{y_column}')]
        plot.add_tools(hover)

        customize_plot(plot)
        plot.legend.title = f'{x_column} vs {y_column}'
        plot.legend.location = "top_left"
        plot.add_layout(Title(text="Scatter Plot", align="center"), "above")

    def bar_chart(data, x_column, y_column, plot):
        x_values = data[x_column]
        y_values = data[y_column]
        colors = viridis(len(x_values))

        plot.vbar(
            x=x_values, top=y_values, width=0.5, color=colors, legend_label=y_column,
            line_color="white", fill_alpha=0.7
        )

        customize_plot(plot)
        plot.legend.title = y_column
        plot.legend.label_text_font_size = "12pt"
        plot.legend.click_policy = "hide"
        plot.xaxis.major_label_orientation = "vertical"
        plot.xaxis.major_label_standoff = 10
        plot.add_layout(Title(text="Bar Chart", align="center"), "above")

        hover = HoverTool()
        hover.tooltips = [(x_column, f'@{x_column}'), (y_column, f'@{y_column}')]
        plot.add_tools(hover)


    def line_chart(data, x_column, y_column, plot):
        plot.line(x=data[x_column], y=data[y_column], line_width=2, color="green", legend_label=y_column)
        plot.circle(x=data[x_column], y=data[y_column], size=8, color="red", legend_label=y_column, fill_alpha=0.5)

        customize_plot(plot)
        plot.legend.title = y_column
        plot.legend.location = "top_left"
        plot.add_layout(Title(text="Line Plot", align="center"), "above")

        hover = HoverTool()
        hover.tooltips = [(x_column, f'@{x_column}'), (y_column, f'@{y_column}')]
        plot.add_tools(hover)


    def histogram_plot(data, x_column, y_column, plot):
        hist, edges = np.histogram(data[y_column], bins=20)
        hist_source = ColumnDataSource(data=dict(top=hist, edges_left=edges[:-1], edges_right=edges[1:]))
        
        plot.quad(top='top', bottom=0, left='edges_left', right='edges_right', source=hist_source,
                fill_color=linear_cmap(field_name='top', palette=Viridis256, low=min(hist), high=max(hist)))

        customize_plot(plot)
        plot.legend.title = x_column
        plot.legend.label_text_font_size = "12pt"
        plot.legend.location = "top_left"
        plot.xaxis.axis_label = x_column
        plot.yaxis.axis_label = 'Frequency'
        plot.title.text = f"Histogram of {y_column.capitalize()}"

        hover = HoverTool()
        hover.tooltips = [("Range", "(@edges_left, @edges_right)"), ("Frequency", "@top")]
        plot.add_tools(hover)

    def interval_chart(data, x_column, y_column, plot):
        source = ColumnDataSource(data=dict(base=data[x_column], lower=data[y_column]-0.2, upper=data[y_column]+0.2))
        plot.segment(x0='base', y0='lower', x1='base', y1='upper', line_width=2, source=source, color="magenta")
        plot.circle(x='base', y='lower', size=6, source=source, color="red", fill_alpha=0.6)
        plot.triangle(x='base', y='upper', size=6, source=source, color="green", fill_alpha=0.6)

        customize_plot(plot)
        plot.legend.title = f'{x_column} vs {y_column}'
        plot.xaxis.major_label_orientation = "vertical"
        plot.xaxis.major_label_standoff = 10
        plot.add_layout(Title(text="Interval Chart", align="center"), "above")
        plot.legend.location = "top_left"

        hover = HoverTool()
        hover.tooltips = [(x_column, f'@{x_column}'), (y_column, f'@{y_column}')]
        plot.add_tools(hover)


    def transform_markers_plot(data, x_column, y_column, plot):
        source = ColumnDataSource(data=dict(x=data[x_column], y=data[y_column]))
        plot.diamond(x='x', y='y', source=source, size=8, color="black", alpha=0.5)
        plot.add_tools(HoverTool(tooltips=[(x_column, '@x'), (y_column, '@y')]))
    

        customize_plot(plot)
        plot.add_layout(Title(text="TransformMarkers", align="center"), "above")
        plot.xaxis.major_label_orientation = "vertical"
        plot.xaxis.major_label_standoff = 10
        plot.legend.location = "top_left" 

        hover = HoverTool()
        hover.tooltips = [(x_column, f'@{x_column}'), (y_column, f'@{y_column}')]
        plot.add_tools(hover)

    #This combine_plot features is not working now
    # Modify generate_combined_plot function to generate a combined plot
    def generate_combined_plot(data, x_column, y_column):
        plots = []

        scatter_plot = generate_custom_plot(data, 'Scatter', x_column, y_column)
        line_plot = generate_custom_plot(data, 'Line', x_column, y_column)
        bar_chart_plot = generate_custom_plot(data, 'Bar', x_column, y_column)
        histogram_plot = generate_custom_plot(data, 'Histogram', x_column, y_column)
        interval_chart_plot = generate_custom_plot(data, 'Interval', x_column, y_column)
        transform_markers_plot = generate_custom_plot(data, 'TransformMarkers', x_column, y_column)

        plots.append([scatter_plot])
        plots.append([line_plot])
        plots.append([bar_chart_plot])
        plots.append([histogram_plot])
        plots.append([interval_chart_plot])
        plots.append([transform_markers_plot])

        combined_layout = gridplot(plots, ncols=2, sizing_mode='scale_both')

        script, div = components(combined_layout)

        hover = HoverTool()
        hover.tooltips = [(x_column, f'@{x_column}'), (y_column, f'@{y_column}')]

        for plot in plots:
            plot[0].add_tools(hover)
            customize_plot(plot[0])

        return script, div



    def customize_plot(plot):
        plot.title.text_font_size = "16pt"
        plot.xaxis.axis_label_text_font_size = "14pt"
        plot.yaxis.axis_label_text_font_size = "14pt"
        plot.xaxis.major_label_text_font_size = "12pt"
        plot.yaxis.major_label_text_font_size = "12pt"
        plot.toolbar.logo = None
        plot.toolbar.tools = [tool for tool in plot.toolbar.tools if not isinstance(tool, HelpTool)]
        plot.toolbar.autohide = True

    # Add this new route at the end of your Flask application
    @app.route('/download_heatmap/<filename>', methods=['GET'])
    def download_heatmap(filename):
        try:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            data = pd.read_csv(file_path)

            # Generate the Seaborn heatmap
            heatmap_image_data = generate_missing_values_heatmap(data)

            # Create a temporary file to store the heatmap image
            temp_file = os.path.join(app.config['UPLOAD_FOLDER'], 'heatmap.png')
            with open(temp_file, 'wb') as f:
                f.write(heatmap_image_data)

            # Send the heatmap image file as a downloadable response
            return send_file(temp_file, as_attachment=True, download_name='heatmap.png', mimetype='image/png')

        except Exception as e:
            logger.error(f"Exception occurred: {str(e)}")
            return make_response(jsonify('Internal error occurred: Please try again'), 500)


    # Function to highlight rows in the missing values table if "Missing Value Count" > 0
    def highlight_missing_values_table_rows(dataframe):
        """
        Highlight rows in the missing values table if "Missing Value Count" > 0.
        """
        highlighted_rows = session.get('highlighted_rows', set())
        
        def highlight_row(index):
            highlighted_rows.add(index)
            session['highlighted_rows'] = highlighted_rows
        
        def unhighlight_row(index):
            highlighted_rows.discard(index)
            session['highlighted_rows'] = highlighted_rows
        
        def row_style(index):
            return 'background-color: yellow' if index in highlighted_rows else ''
        
        # Apply styles to the DataFrame
        styled_dataframe = dataframe.style.applymap(lambda _: row_style(_.name), subset=pd.IndexSlice[highlighted_rows, :])
        
        # Add JavaScript to handle double-click events for removing highlights
        styled_dataframe.add_class('highlight-rows')
        styled_dataframe.set_table_styles([
            dict(selector='.highlight-rows', props=[('cursor', 'pointer')]),
        ])
        styled_dataframe = styled_dataframe.set_properties(**{
            'onmouseover': f'highlight_row({{index}})',
            'onmouseout': f'unhighlight_row({{index}})',
            'ondblclick': f'unhighlight_row({{index}})',
        })

        return styled_dataframe


    @app.route('/reset_highlighted_rows', methods=['POST'])
    def reset_highlighted_rows():
        session['highlighted_rows'] = None  # Clear the highlighted rows
        return redirect(url_for('analyze', filename=session.get('filename')))


    # Function to split numeric values from alphabetical values and remove alphabetical characters when joining with numeric values.
    def split_and_clean(data, selected_column, clean_all_columns=False):
        """
        Split numeric values from alphabetical values and drop columns with mixed data types.

        Parameters:
        - data (pd.DataFrame): The DataFrame containing the data.
        - selected_column (str): The name of the column selected for split_and_clean.
        - clean_all_columns (bool): Whether to clean all columns.

        Returns:
        - data_cleaned (pd.DataFrame): The DataFrame with cleaned and split values.
        """
        data_cleaned = data.copy()

        if clean_all_columns:
            for column in data_cleaned.columns:
                if data_cleaned[column].dtype == 'object':
                    data_cleaned[column] = data_cleaned[column].apply(lambda x: re.sub(r'[^0-9]+', '', str(x)) if any(c.isdigit() for c in str(x)) else x)
        else:
            if data_cleaned[selected_column].dtype == 'object':
                data_cleaned[selected_column] = data_cleaned[selected_column].apply(lambda x: re.sub(r'[^0-9]+', '', str(x)) if any(c.isdigit() for c in str(x)) else x)

        # Drop columns that contain both numerical and text values in any row
        mixed_data_cols = []
        for column in data_cleaned.columns:
            if data_cleaned[column].apply(lambda x: isinstance(x, (int, float))).any() and data_cleaned[column].apply(lambda x: isinstance(x, str)).any():
                mixed_data_cols.append(column)

        data_cleaned = data_cleaned.drop(columns=mixed_data_cols)

        return data_cleaned

    # Function to highlight missing value rows
    def highlight_missing_values_rows(dataframe):
        mask = dataframe.isnull().any(axis=1)  # Create a mask for rows with missing values
        return dataframe[mask].style.apply(lambda _: 'background-color: yellow', axis=None)

    return app

def setup_logging():
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')

    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

app = create_app()

if __name__ == '__main__':
    logging.basicConfig(filename='app.log', level=logging.INFO)
    app.run(debug=True, port=1998)