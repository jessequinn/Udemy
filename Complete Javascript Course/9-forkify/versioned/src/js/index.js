import Search from './models/Search';
import Recipe from './models/Recipe';
import List from './models/List';
import Likes from './models/Likes';
import * as searchView from './views/searchView';
import * as recipeView from './views/recipeView';
import * as listView from './views/listView';
import * as likesView from './views/likesView';
import {clearLoader, elements, renderLoader} from "./views/base";

/** global state of the app
 * - search object
 * - current recipe object
 * - shopping list object
 * - liked recipes
 */
const state = {};
// testing
// window.state = state;

/**
 * SEARCH CONTROLLER
 */
const controlSearch = async () => {
    // const query = 'pizza'; // TODO: query string
    const query = searchView.getInput();

    if (query) {
        state.search = new Search(query);

        searchView.clearInput();
        searchView.clearResults();
        renderLoader(elements.searchResult);

        try {
            await state.search.getResults();

            // console.log(state.search.result);
            clearLoader();
            searchView.renderResults(state.search.result)
        } catch (e) {
            throw Error(e);
            clearLoader();
        }
    }
};

elements.searchForm.addEventListener('submit', e => {
    e.preventDefault();

    controlSearch();
});

/**
 * TESTING PURPOSES
 */
// window.addEventListener('load', e => {
//     e.preventDefault();
//
//     controlSearch();
// });

elements.searchResultPages.addEventListener('click', e => {
    const btn = e.target.closest('.btn-inline');

    if (btn) {
        const goToPage = parseInt(btn.dataset.goto, 10);
        searchView.clearResults();
        searchView.renderResults(state.search.result, goToPage);
    }
});

/**
 * RECIPE CONTROLLER
 */
const controlRecipe = async () => {
    const id = window.location.hash.replace('#', '');

    if (id) {
        recipeView.clearRecipe();
        renderLoader(elements.recipe);

        if (state.search) searchView.highlightSelected(id);

        state.recipe = new Recipe(id);

        // TESTING
        // window.r = state.recipe;

        try {
            await state.recipe.getRecipe();
            state.recipe.parseIngredients();
            // console.log(state.recipe.ingredients);

            state.recipe.calcTime();
            state.recipe.calcServings();


            // console.log(state.recipe);
            clearLoader();
            recipeView.renderRecipe(state.recipe, state.likes.isLiked(id));

        } catch (e) {
            throw Error(e);
        }
    }
};

// window.addEventListener('hashchange', controlRecipe);
// window.addEventListener('load', controlRecipe);
['hashchange', 'load'].forEach(e => window.addEventListener(e, controlRecipe));

/**
 * LIST CONTROLLER
 */
const controlList = () => {
    if (!state.list) state.list = new List();

    state.recipe.ingredients.forEach(el => {
        const item = state.list.addItem(el.count, el.unit, el.ingredient);
        listView.renderItem(item);
    });
};

/**
 * LIKE CONTROLLER
 */
// testing
// state.likes = new Likes();

const controlLike = () => {
    if (!state.likes) state.likes = new Likes();

    const currentId = state.recipe.id;
    if(!state.likes.isLiked(currentId)) {
        const newLike = state.likes.addLike(currentId, state.recipe.title, state.recipe.author, state.recipe.img);

        likesView.toggleLikeBtn(true);
        likesView.renderLike(newLike);
    } else {
        state.likes.deleteLike(currentId);
        likesView.toggleLikeBtn(false);
        likesView.deleteLike(currentId);
    }

    likesView.toggleLikeMenu(state.likes.getNumberOfLikes());
};

elements.shopping.addEventListener('click', e => {
    const id = e.target.closest('.shopping__item').dataset.itemid;

    if (e.target.matches('.shopping__delete, .shopping__delete *')) {
        state.list.deleteItem(id);
        listView.deleteItem(id);
    } else if (e.target.matches('.shopping__count-value')) {
        const val = parseInt(e.target.value, 10);

        state.list.updateCount(id, val);
    }
});

window.addEventListener('load', e => {
    state.likes = new Likes();

    state.likes.readStorage();
    likesView.toggleLikeMenu(state.likes.getNumberOfLikes());
    state.likes.likes.forEach(like => likesView.renderLike(like));
});


/**
 * recipe handling button clicks
 */
elements.recipe.addEventListener('click', e => {
    if (e.target.matches('.btn-decrease, .btn-decrease *')) {
        if (state.recipe.servings > 1) {
            state.recipe.updateServings('dec');
            recipeView.updateServingsIngredients(state.recipe);
        }
    } else if (e.target.matches('.btn-increase, .btn-increase *')) {
        state.recipe.updateServings('inc');
        recipeView.updateServingsIngredients(state.recipe);
    } else if (e.target.matches('.recipe__btn--add, .recipe__btn--add *')) {
        controlList();
    } else if (e.target.matches('.recipe__love, .recipe__love *')) {
        controlLike();
    }
});

// window.l = new List();