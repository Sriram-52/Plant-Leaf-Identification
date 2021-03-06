from flask import Flask, request, render_template

# tensor flow imports
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession

# keras imports
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# general imports
import numpy as np
import wikipedia
from os import listdir

# GUI related code to compress total usage
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# initalise app
app = Flask(__name__)

# load the trained model
model_path = "Model/model_inception.h5"
model = load_model(model_path)

# plants directories
dirs = ['abies_concolor', 'abies_nordmanniana', 'acer_campestre', 'acer_ginnala', 'acer_griseum', 'acer_negundo', 'acer_palmatum', 'acer_pensylvanicum', 'acer_platanoides', 'acer_pseudoplatanus', 'acer_rubrum', 'acer_saccharinum', 'acer_saccharum', 'aesculus_flava', 'aesculus_glabra', 'aesculus_hippocastamon', 'aesculus_pavi', 'ailanthus_altissima', 'albizia_julibrissin', 'amelanchier_arborea', 'amelanchier_canadensis', 'amelanchier_laevis', 'asimina_triloba', 'betula_alleghaniensis', 'betula_jacqemontii', 'betula_lenta', 'betula_nigra', 'betula_populifolia', 'broussonettia_papyrifera', 'carpinus_betulus', 'carpinus_caroliniana', 'carya_cordiformis', 'carya_glabra', 'carya_ovata', 'carya_tomentosa', 'castanea_dentata', 'catalpa_bignonioides', 'catalpa_speciosa', 'cedrus_atlantica', 'cedrus_deodara', 'cedrus_libani', 'celtis_occidentalis', 'celtis_tenuifolia', 'cercidiphyllum_japonicum', 'cercis_canadensis', 'chamaecyparis_pisifera', 'chamaecyparis_thyoides', 'chionanthus_retusus', 'chionanthus_virginicus', 'cladrastis_lutea', 'cornus_florida', 'cornus_kousa', 'cornus_mas', 'corylus_colurna', 'crataegus_crus-galli', 'crataegus_laevigata', 'crataegus_phaenopyrum', 'crataegus_pruinosa', 'crataegus_viridis', 'cryptomeria_japonica', 'diospyros_virginiana', 'eucommia_ulmoides', 'evodia_daniellii', 'fagus_grandifolia', 'ficus_carica', 'fraxinus_americana', 'fraxinus_nigra', 'fraxinus_pennsylvanica', 'ginkgo_biloba', 'gleditsia_triacanthos', 'gymnocladus_dioicus', 'halesia_tetraptera', 'ilex_opaca', 'juglans_cinerea', 'juglans_nigra', 'juniperus_virginiana', 'koelreuteria_paniculata', 'larix_decidua', 'liquidambar_styraciflua', 'liriodendron_tulipifera', 'maclura_pomifera', 'magnolia_acuminata', 'magnolia_denudata', 'magnolia_grandiflora', 'magnolia_macrophylla', 'magnolia_soulangiana', 'magnolia_stellata', 'magnolia_tripetala', 'magnolia_virginiana',
        'malus_angustifolia', 'malus_baccata', 'malus_coronaria', 'malus_floribunda', 'malus_hupehensis', 'malus_pumila', 'metasequoia_glyptostroboides', 'morus_alba', 'morus_rubra', 'nyssa_sylvatica', 'ostrya_virginiana', 'oxydendrum_arboreum', 'paulownia_tomentosa', 'phellodendron_amurense', 'picea_abies', 'picea_orientalis', 'picea_pungens', 'pinus_bungeana', 'pinus_cembra', 'pinus_densiflora', 'pinus_echinata', 'pinus_flexilis', 'pinus_koraiensis', 'pinus_nigra', 'pinus_parviflora', 'pinus_peucea', 'pinus_pungens', 'pinus_resinosa', 'pinus_rigida', 'pinus_strobus', 'pinus_sylvestris', 'pinus_taeda', 'pinus_thunbergii', 'pinus_virginiana', 'pinus_wallichiana', 'platanus_acerifolia', 'platanus_occidentalis', 'populus_deltoides', 'populus_grandidentata', 'populus_tremuloides', 'prunus_pensylvanica', 'prunus_sargentii', 'prunus_serotina', 'prunus_serrulata', 'prunus_subhirtella', 'prunus_virginiana', 'prunus_yedoensis', 'pseudolarix_amabilis', 'ptelea_trifoliata', 'pyrus_calleryana', 'quercus_acutissima', 'quercus_alba', 'quercus_bicolor', 'quercus_cerris', 'quercus_coccinea', 'quercus_falcata', 'quercus_imbricaria', 'quercus_macrocarpa', 'quercus_marilandica', 'quercus_michauxii', 'quercus_montana', 'quercus_muehlenbergii', 'quercus_nigra', 'quercus_palustris', 'quercus_phellos', 'quercus_robur', 'quercus_rubra', 'quercus_shumardii', 'quercus_stellata', 'quercus_velutina', 'quercus_virginiana', 'robinia_pseudo-acacia', 'salix_babylonica', 'salix_caroliniana', 'salix_matsudana', 'salix_nigra', 'sassafras_albidum', 'staphylea_trifolia', 'stewartia_pseudocamellia', 'styrax_japonica', 'styrax_obassia', 'syringa_reticulata', 'taxodium_distichum', 'tilia_americana', 'tilia_cordata', 'tilia_europaea', 'tilia_tomentosa', 'toona_sinensis', 'tsuga_canadensis', 'ulmus_americana', 'ulmus_glabra', 'ulmus_parvifolia', 'ulmus_pumila', 'ulmus_rubra', 'zelkova_serrata']


def predict_image(path):
    img = image.load_img(path, target_size=(224, 224))
    image_arr = image.img_to_array(img)
    image_arr = image_arr / 255  # normalising image
    image_arr = np.expand_dims(image_arr, axis=0)
    prediction = model.predict(image_arr)
    prediction = np.argmax(prediction, axis=1)
    result = dirs[prediction[0]]
    return wikipedia.summary(result, sentences=2)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == "POST":
        file = request.files['file']
        file_path = 'Uploads/' + file.filename
        file.save(file_path)
        prediction = predict_image(file_path)
        return prediction

    return None


if __name__ == "__main__":
    app.run(port=5000, debug=True)
