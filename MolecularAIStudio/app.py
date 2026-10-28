
import streamlit as st
import pandas as pd
import py3Dmol
import streamlit.components.v1 as components

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Draw, AllChem
import base64
from pathlib import Path


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Molecular AI Studio",
    page_icon="🧪",
    layout="wide"
)
# =========================================================
# PAGE BACKGROUND FUNCTION
# =========================================================

def set_page_background(image_name):
    image_path = Path(__file__).parent / "assets" / image_name

    if not image_path.is_file():
        st.warning(f"Background image not found: {image_name}")
        return

    encoded_image = base64.b64encode(
        image_path.read_bytes()
    ).decode("utf-8")

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(
                    rgba(3, 13, 30, 0.60),
                    rgba(3, 13, 30, 0.72)
                ),
                url("data:image/png;base64,{encoded_image}");

            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}

        section[data-testid="stSidebar"] {{
            background: rgba(3, 13, 30, 0.96);
            border-right: 1px solid rgba(0, 180, 255, 0.35);
        }}

        h1, h2, h3 {{
            color: #eaf7ff;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 2. MOLECULE LIBRARY
# =========================================================

molecules = {
    "Ethanol": "CCO",
    "Water": "O",
    "Methane": "C",
    "Methanol": "CO",
    "Acetic Acid": "CC(=O)O",
    "Benzene": "c1ccccc1",
    "Glucose": "C(C1C(C(C(C(O1)O)O)O)O)O",
    "Acetone": "CC(=O)C",
    "Caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    "Lactic Acid": "CC(O)C(=O)O",
    "Glycine": "NCC(=O)O",
    "Cyclohexane": "C1CCCCC1",
    "Toluene": "Cc1ccccc1",
    "Urea": "NC(=O)N",
    "Formaldehyde": "C=O",
    "Ammonia": "N",
    "Vitamin C (Ascorbic Acid)": "O=C1OC(C(O)CO)C(O)=C1O"
}


# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================

def calculate_properties(mol):
    """Calculate basic molecular descriptors using RDKit."""

    return {
        "Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular Weight": round(Descriptors.MolWt(mol), 3),
        "LogP": round(Descriptors.MolLogP(mol), 3),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 3),
        "H-Bond Donors": rdMolDescriptors.CalcNumHBD(mol),
        "H-Bond Acceptors": rdMolDescriptors.CalcNumHBA(mol),
        "Rotatable Bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "Heavy Atoms": mol.GetNumHeavyAtoms(),
        "Ring Count": rdMolDescriptors.CalcNumRings(mol)
    }


def generate_3d_molecule(smiles, show_hydrogens=True):
    """Create and optimize a 3D molecule from a SMILES string."""

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None, "Invalid SMILES string."

    # Add hydrogens before generating 3D coordinates.
    mol = Chem.AddHs(mol)

    # Generate initial 3D coordinates.
    result = AllChem.EmbedMolecule(
        mol,
        randomSeed=42
    )

    if result == -1:
        # Retry with a different embedding approach.
        result = AllChem.EmbedMolecule(
            mol,
            randomSeed=123,
            useRandomCoords=True
        )

    if result == -1:
        return None, "RDKit could not generate 3D coordinates."

    # Optimize the geometry using an available force field.
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(
                mol,
                maxIters=200
            )
        else:
            AllChem.UFFOptimizeMolecule(
                mol,
                maxIters=200
            )
    except Exception as error:
        return None, f"Geometry optimization failed: {error}"

    # Remove hydrogens only after generating and optimizing
    # the 3D geometry, if the user does not want to display them.
    if not show_hydrogens:
        mol = Chem.RemoveHs(mol)

    return mol, None


def show_3d_viewer(mol, display_style):
    """Render a molecule using the selected py3Dmol style."""

    mol_block = Chem.MolToMolBlock(mol)

    viewer = py3Dmol.view(
        width=800,
        height=500
    )

    viewer.addModel(mol_block, "mol")

    if display_style == "Ball and Stick":
        viewer.setStyle({
            "stick": {
                "radius": 0.18,
                "colorscheme": "Jmol"
            },
            "sphere": {
                "scale": 0.30,
                "colorscheme": "Jmol"
            }
        })

    elif display_style == "Stick":
        viewer.setStyle({
            "stick": {
                "radius": 0.22,
                "colorscheme": "Jmol"
            }
        })

    elif display_style == "Space Filling":
        viewer.setStyle({
            "sphere": {
                "scale": 0.9,
                "colorscheme": "Jmol"
            }
        })

    elif display_style == "Wireframe":
        viewer.setStyle({
            "line": {
                "linewidth": 1,
                "colorscheme": "Jmol"
            }
        })

    viewer.setBackgroundColor("#FFFFFF")
    viewer.zoomTo()

    components.html(
        viewer._make_html(),
        height=520,
        scrolling=False
    )


# =========================================================
# 4. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.title("🧪 Molecular AI")
    st.caption("Molecular Analysis Workbench")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🧬 Single Molecule",
            "📊 Batch Analysis",
            "🧊 3D Viewer",
           
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption("Powered by RDKit and Streamlit")
    # =========================================================
# APPLY DIFFERENT BACKGROUND TO EACH PAGE
# =========================================================

page_backgrounds = {
    "🏠 Dashboard": "dashboard_lab.png",
    "🧬 Single Molecule": "molecular_pattern.png",
    "📊 Batch Analysis": "molecule_closeup.png",
    "🧊 3D Viewer": "holographic_lab.png"
}

set_page_background(page_backgrounds[page])


# =========================================================
# 5. DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.title("🧪 Molecular AI Studio")
    st.subheader("Welcome to My Chemistry Project")

    st.markdown("""
    ### 👨‍🎓 Developed by

    **PRAGADEESWARAN T**

    Student

    SRM Institute of Science and Technology, Tiruchirappalli

    **For queries, contact:** pragadeeswaran1324@gmail.com
    """)

    st.divider()

    st.header("Project Overview")

    st.write(
        "Molecular AI Studio is a chemistry application for "
        "exploring molecular structures, calculating molecular "
        "descriptors, analyzing SMILES, and visualizing molecules."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Molecule Library",
        len(molecules)
    )

    col2.metric(
        "Chemistry Engine",
        "RDKit"
    )

    col3.metric(
        "AI Prediction",
        "In Development"
    )

    st.divider()

    st.header("Available Tools")

    tool_col1, tool_col2 = st.columns(2)

    with tool_col1:
        st.markdown("""
        - 🧬 Single-molecule analysis
        - 📊 Batch SMILES analysis
        - 🧊 Interactive 3D visualization
        """)

    with tool_col2:
        st.markdown("""
        - 🧪 Molecular property calculations
        """)

    st.info(
        "AI prediction and molecule generation require additional "
        "models and are not yet implemented in this version."
    )


# =========================================================
# 6. SINGLE MOLECULE ANALYSIS
# =========================================================

elif page == "🧬 Single Molecule":

    st.title("🧬 Single Molecule Analysis")
    st.write("Choose a molecule or enter your own SMILES string.")

    col_search, col_select = st.columns(2)

    with col_search:
        search_smiles = st.text_input(
            "Enter SMILES",
            placeholder="Example: CC(=O)O",
            key="single_smiles"
        )

    with col_select:
        selected_name = st.selectbox(
            "Choose a molecule",
            list(molecules.keys()),
            key="single_molecule"
        )

    if search_smiles.strip():
        smiles = search_smiles.strip()
        molecule_name = "Custom molecule"
    else:
        smiles = molecules[selected_name]
        molecule_name = selected_name

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        st.error(
            "Invalid SMILES string. Check the input and try again."
        )

    else:
        st.success(f"Valid structure: {molecule_name}")

        left, right = st.columns(2)

        with left:
            st.subheader("2D Molecular Structure")

            image = Draw.MolToImage(
                mol,
                size=(450, 350)
            )

            st.image(
                image,
                caption=molecule_name,
                use_container_width=True
            )

        with right:
            st.subheader("Molecular Information")

            st.write(f"**Name:** {molecule_name}")
            st.code(smiles, language=None)

            properties = calculate_properties(mol)

            st.write(f"**Formula:** {properties['Formula']}")
            st.write(
                f"**Molecular Weight:** "
                f"{properties['Molecular Weight']} g/mol"
            )
            st.write(f"**LogP:** {properties['LogP']}")
            st.write(f"**TPSA:** {properties['TPSA']} Å²")
            st.write(
                f"**Hydrogen Bond Donors:** "
                f"{properties['H-Bond Donors']}"
            )
            st.write(
                f"**Hydrogen Bond Acceptors:** "
                f"{properties['H-Bond Acceptors']}"
            )
            st.write(
                f"**Rotatable Bonds:** "
                f"{properties['Rotatable Bonds']}"
            )
            st.write(f"**Heavy Atoms:** {properties['Heavy Atoms']}")
            st.write(f"**Ring Count:** {properties['Ring Count']}")

        report = pd.DataFrame(
            [{"Molecule": molecule_name, "SMILES": smiles, **properties}]
        )

        st.download_button(
            "Download Molecule Report (CSV)",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="molecule_report.csv",
            mime="text/csv"
        )


# =========================================================
# 7. BATCH MOLECULE ANALYSIS
# =========================================================

elif page == "📊 Batch Analysis":

    st.title("📊 Batch Molecule Analysis")
    st.write("Enter one SMILES string per line.")

    examples = "CCO\nCC(=O)O\nc1ccccc1"

    text = st.text_area(
        "Molecule list",
        value=examples,
        height=160
    )

    if st.button("Analyze Molecules"):

        rows = []

        for line in text.splitlines():

            smiles_value = line.strip()

            if not smiles_value:
                continue

            mol = Chem.MolFromSmiles(smiles_value)

            if mol is None:
                rows.append({
                    "SMILES": smiles_value,
                    "Status": "Invalid"
                })
                continue

            rows.append({
                "SMILES": smiles_value,
                "Status": "Valid",
                **calculate_properties(mol)
            })

        if rows:
            df = pd.DataFrame(rows)

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Results as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="molecule_properties.csv",
                mime="text/csv"
            )

            valid_count = sum(
                row["Status"] == "Valid" for row in rows
            )

            invalid_count = len(rows) - valid_count

            st.success(
                f"Processed {len(rows)} entries: "
                f"{valid_count} valid, {invalid_count} invalid."
            )

        else:
            st.warning("Please enter at least one SMILES string.")


# =========================================================
# 8. INTERACTIVE 3D MOLECULE VIEWER
# =========================================================

elif page == "🧊 3D Viewer":

    st.title("🧊 Interactive 3D Molecule Viewer")

    st.write(
        "Generate a 3D molecular geometry and explore it "
        "using different visualization styles."
    )

    col1, col2 = st.columns(2)

    with col1:
        viewer_name = st.selectbox(
            "Choose a molecule",
            list(molecules.keys()),
            key="viewer_molecule"
        )

    with col2:
        display_style = st.selectbox(
            "Visualization style",
            [
                "Ball and Stick",
                "Stick",
                "Space Filling",
                "Wireframe"
            ],
            key="viewer_style"
        )

    custom_3d_smiles = st.text_input(
        "Or enter a custom SMILES string (optional)",
        placeholder="Example: CCO",
        key="custom_3d_smiles"
    )

    show_hydrogens = st.checkbox(
        "Show hydrogen atoms",
        value=True,
        key="viewer_hydrogens"
    )

    if custom_3d_smiles.strip():
        smiles_3d = custom_3d_smiles.strip()
        name_3d = "Custom molecule"
    else:
        smiles_3d = molecules[viewer_name]
        name_3d = viewer_name

    if st.button("Generate 3D Structure", type="primary"):

        with st.spinner("Generating and optimizing 3D geometry..."):

            mol_3d, error_message = generate_3d_molecule(
                smiles_3d,
                show_hydrogens=show_hydrogens
            )

            if error_message:
                st.error(error_message)

            elif mol_3d is not None:

                st.subheader(f"3D Structure: {name_3d}")

                try:
                    show_3d_viewer(
                        mol_3d,
                        display_style
                    )

                    st.success(
                        f"3D structure generated using "
                        f"the {display_style} style."
                    )

                    st.caption(
                        "Rotate the molecule by dragging it, "
                        "and zoom using your mouse wheel. "
                        "This is a computed molecular geometry, "
                        "not an experimentally determined structure."
                    )

                    mol_block = Chem.MolToMolBlock(mol_3d)

                    st.download_button(
                        "Download 3D Structure (.mol)",
                        data=mol_block,
                        file_name="molecule_3d.mol",
                        mime="chemical/x-mdl-molfile"
                    )

                except Exception as error:
                    st.error(
                        f"Unable to display the 3D structure: {error}"
                    )
                    
# =========================================================
# 9. FOOTER
# =========================================================

st.divider()

st.caption(
    "Molecular AI Studio | Molecular descriptors calculated "
    "using RDKit. Calculated descriptors are approximate and may vary based on the"
)
