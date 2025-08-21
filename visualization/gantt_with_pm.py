
import matplotlib.pyplot as plt
from .color_scheme import create_colormap

def plot(job_shop, pm_schedule):
    """
    Plots a Gantt chart including both production operations and preventive maintenance.

    Args:
        job_shop (JobShop): The JobShop object containing the production schedule.
        pm_schedule (list): A list of tuples, where each tuple represents a PM activity
                              in the format (machine_id, start_time, duration).
    """
    fig, ax = plt.subplots()
    colormap = create_colormap()

    # 1. Plot production operations from the JobShop object
    for machine in job_shop.machines:
        # Sort operations by start time for correct plotting order
        machine_operations = sorted(machine._processed_operations, key=lambda op: op.scheduling_information['start_time'])
        
        for operation in machine_operations:
            operation_start = operation.scheduling_information['start_time']
            operation_end = operation.scheduling_information['end_time']
            operation_duration = operation_end - operation_start
            operation_label = f"{operation.operation_id}"

            # Set color based on job ID
            color_index = operation.job_id % len(job_shop.jobs)
            if color_index >= colormap.N:
                color_index = color_index % colormap.N
            color = colormap(color_index)

            # Draw the operation bar
            ax.broken_barh(
                [(operation_start, operation_duration)],
                (machine.machine_id - 0.4, 0.8),
                facecolors=color,
                edgecolor='black'
            )

            # Draw setup time if it exists
            setup_start = operation.scheduling_information.get('start_setup')
            setup_time = operation.scheduling_information.get('setup_time')
            if setup_time and setup_start is not None:
                ax.broken_barh(
                    [(setup_start, setup_time)],
                    (machine.machine_id - 0.4, 0.8),
                    facecolors='grey',
                    edgecolor='black', hatch='/')
            
            # Add operation label in the middle of the bar
            middle_of_operation = operation_start + operation_duration / 2
            ax.text(
                middle_of_operation,
                machine.machine_id,
                operation_label,
                ha='center',
                va='center',
                fontsize=8
            )

    # 2. Plot PM activities
    for pm_activity in pm_schedule:
        machine_id, pm_start, pm_duration = pm_activity
        
        # Draw the PM bar with a distinct style (grey with 'xx' hatch)
        ax.broken_barh(
            [(pm_start, pm_duration)],
            (machine_id - 0.4, 0.8),
            facecolors='#808080',  # Grey color
            edgecolor='black',
            hatch='xx'
        )
        
        # Add "PM" label in the middle of the bar
        middle_of_pm = pm_start + pm_duration / 2
        ax.text(
            middle_of_pm,
            machine_id,
            'PM',
            ha='center',
            va='center',
            fontsize=8,
            color='white',
            fontweight='bold'
        )

        # Add a marker at the PM start time to indicate risk reset
        ax.plot(pm_start, machine_id, 'D', color='red', markersize=8, markeredgecolor='black', zorder=10) # D for diamond

    # 3. Configure plot aesthetics
    fig.set_size_inches(15, 6)  # A slightly wider figure to accommodate more details

    ax.set_yticks(range(job_shop.nr_of_machines))
    ax.set_yticklabels([f'M{machine_id+1}' for machine_id in range(job_shop.nr_of_machines)])
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_title('Gantt Chart with Maintenance Schedule')
    ax.grid(True)

    return plt
