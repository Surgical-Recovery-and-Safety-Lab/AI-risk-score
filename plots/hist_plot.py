def plt_hist(operation_years):
    # 2. Create the plot
    plt.figure(dpi=300)

    # Define bins to align with calendar years
    bins = np.arange(min(operation_years), max(operation_years) + 2) - 0.5

    # Plotting the histogram
    # 'alpha' controls transparency (0 = fully transparent, 1 = opaque)
    # 'edgecolor' is recommended so bins remain distinct when transparent
    colours = plt.get_cmap("Pastel1")
    plt.hist(
        operation_years,
        bins=bins,
        color=colours(0),
        edgecolor="black",
        linewidth=1.2,
        label="Patient Count",
    )

    # 3. Add labels and styling for a research paper
    plt.xlabel("Operation year", fontweight="bold")
    plt.ylabel("Number of patients", fontweight="bold")

    # Ensure x-axis shows individual years or intervals
    plt.xticks(np.arange(min(operation_years), max(operation_years) + 1, 2))

    # Add a light grid to demonstrate the transparency effect
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    # Remove top and right spines for a cleaner look
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()

    # 4. Save the figure
    plt.savefig("patient_distribution_histogram.png", dpi=300)
    plt.show()
